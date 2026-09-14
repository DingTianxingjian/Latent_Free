"""Minimal stand-in for tensorflow_graphics.geometry.transformation used by three UniVLA
transforms (furniture_bench, iamlab_cmu_pickup_insert, ...): euler.from_quaternion.
Same convention as tensorflow_graphics: quaternion (x, y, z, w) -> euler (roll, pitch, yaw),
i.e. rotations about x, y, z; pitch clamped to +-pi/2 at the gimbal-lock boundary."""
import tensorflow as tf

class euler:
    @staticmethod
    def from_quaternion(q):
        q = tf.convert_to_tensor(q, tf.float32)
        x, y, z, w = tf.unstack(q, axis=-1)
        roll = tf.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = tf.clip_by_value(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = tf.asin(sinp)
        yaw = tf.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return tf.stack([roll, pitch, yaw], axis=-1)

# Self-check at import: 90 deg about z -> yaw pi/2; small rotation about x -> roll only.
# The UniVLA transforms import this module INSIDE the traced tf.data function, so the check must
# run eagerly even then (init_scope lifts it out of the graph being built).
with tf.init_scope():
    _q = tf.constant([[0.0, 0.0, 0.70710678, 0.70710678], [0.14943813, 0.0, 0.0, 0.98877108]], tf.float32)
    _e = euler.from_quaternion(_q).numpy()
    assert abs(_e[0, 2] - 1.5707963) < 1e-4 and abs(_e[0, 0]) < 1e-4 and abs(_e[0, 1]) < 1e-4, _e
    assert abs(_e[1, 0] - 0.3) < 1e-4 and abs(_e[1, 1]) < 1e-4 and abs(_e[1, 2]) < 1e-4, _e
