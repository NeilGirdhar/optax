# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for methods in `optax.projections.py`."""

from functools import partial  # pylint: disable=g-importing-member

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax import projections as proj
import optax.tree


def projection_simplex_jacobian(projection):
  """Theoretical expression for the Jacobian of projection_simplex."""
  support = (projection > 0).astype(jnp.int32)
  cardinality = jnp.count_nonzero(support)
  return jnp.diag(support) - jnp.outer(support, support) / cardinality


class ProjectionsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    array_1d = jnp.array([0.5, 2.1, -3.5])
    array_2d = jnp.array([[0.5, 2.1, -3.5], [1.0, 2.0, 3.0]])
    tree = (array_1d, array_1d)
    self.data = {
        'array_1d': array_1d,
        'array_2d': array_2d,
        'tree': tree,
    }
    self.fns = {
        'l1': (proj.projection_l1_ball, partial(optax.tree.norm, ord=1)),
        'l2': (proj.projection_l2_ball, optax.tree.norm),
        'linf': (proj.projection_linf_ball,
                 partial(optax.tree.norm, ord='inf')),
    }

  def test_projection_non_negative(self):
    with self.subTest('with an array'):
      x = jnp.array([-1.0, 2.0, 3.0])
      expected = jnp.array([0, 2.0, 3.0])
      np.testing.assert_array_equal(proj.projection_non_negative(x), expected)

    with self.subTest('with a tuple'):
      np.testing.assert_array_equal(
          proj.projection_non_negative((x, x)), (expected, expected)
      )

    with self.subTest('with nested pytree'):
      tree_x = (-1.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
      tree_expected = (0.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
      chex.assert_trees_all_equal(
          proj.projection_non_negative(tree_x), tree_expected
      )

  def test_projection_box(self):
    with self.subTest('lower and upper are scalars'):
      lower, upper = 0.0, 2.0
      x = jnp.array([-1.0, 2.0, 3.0])
      expected = jnp.array([0, 2.0, 2.0])
      np.testing.assert_array_equal(
          proj.projection_box(x, lower, upper), expected
      )

    with self.subTest('lower and upper values are arrays'):
      lower_arr = jnp.ones(len(x)) * lower
      upper_arr = jnp.ones(len(x)) * upper
      np.testing.assert_array_equal(
          proj.projection_box(x, lower_arr, upper_arr), expected
      )

    with self.subTest('lower and upper are tuples of arrays'):
      lower_tuple = (lower, lower)
      upper_tuple = (upper, upper)
      chex.assert_trees_all_equal(
          proj.projection_box((x, x), lower_tuple, upper_tuple),
          (expected, expected),
      )

    with self.subTest('lower and upper are pytrees'):
      tree = (-1.0, {'k1': 2.0, 'k2': (2.0, 3.0)}, 3.0)
      expected = (0.0, {'k1': 2.0, 'k2': (2.0, 2.0)}, 2.0)
      lower_tree = (0.0, {'k1': 0.0, 'k2': (0.0, 0.0)}, 0.0)
      upper_tree = (2.0, {'k1': 2.0, 'k2': (2.0, 2.0)}, 2.0)
      chex.assert_trees_all_equal(
          proj.projection_box(tree, lower_tree, upper_tree), expected
      )

  def test_projection_hypercube(self):
    x = jnp.array([-1.0, 2.0, 0.5])

    with self.subTest('with default scale'):
      expected = jnp.array([0, 1.0, 0.5])
      np.testing.assert_array_equal(proj.projection_hypercube(x), expected)

    with self.subTest('with scalar scale'):
      expected = jnp.array([0, 0.8, 0.5])
      np.testing.assert_array_equal(proj.projection_hypercube(x, 0.8), expected)

    with self.subTest('with array scales'):
      scales = jnp.ones(len(x)) * 0.8
      np.testing.assert_array_equal(
          proj.projection_hypercube(x, scales), expected
      )

  @parameterized.parameters(1.0, 0.8)
  def test_projection_simplex_array(self, scale):
    rng = np.random.RandomState(0)
    x = rng.randn(50).astype(np.float32)
    p = proj.projection_simplex(x, scale)

    np.testing.assert_almost_equal(jnp.sum(p), scale, decimal=4)
    self.assertTrue(jnp.all(0 <= p))
    self.assertTrue(jnp.all(p <= scale))

  @parameterized.parameters(1.0, 0.8)
  def test_projection_simplex_pytree(self, scale):
    pytree = {'w': jnp.array([2.5, 3.2]), 'b': 0.5}
    new_pytree = proj.projection_simplex(pytree, scale)
    np.testing.assert_almost_equal(optax.tree.sum(new_pytree), scale, decimal=4)

  @parameterized.parameters(1.0, 0.8)
  def test_projection_simplex_edge_case(self, scale):
    p = proj.projection_simplex(jnp.array([0.0, 0.0, -jnp.inf]), scale)
    np.testing.assert_array_almost_equal(
        p, jnp.array([scale / 2, scale / 2, 0.0])
    )

  def test_projection_simplex_jacobian(self):
    rng = np.random.RandomState(0)

    x = rng.rand(5).astype(np.float32)
    v = rng.randn(5).astype(np.float32)

    jac_rev = jax.jacrev(proj.projection_simplex)(x)
    jac_fwd = jax.jacfwd(proj.projection_simplex)(x)

    with self.subTest('Check against theoretical expression'):
      p = proj.projection_simplex(x)
      jac_true = projection_simplex_jacobian(p)

      np.testing.assert_array_almost_equal(jac_true, jac_fwd)
      np.testing.assert_array_almost_equal(jac_true, jac_rev)

    with self.subTest('Check against finite difference'):
      jvp = jax.jvp(proj.projection_simplex, (x,), (v,))[1]
      eps = 1e-4
      jvp_finite_diff = (
          proj.projection_simplex(x + eps * v)
          - proj.projection_simplex(x - eps * v)
      ) / (2 * eps)
      np.testing.assert_array_almost_equal(jvp, jvp_finite_diff, decimal=3)

    with self.subTest('Check vector-Jacobian product'):
      (vjp,) = jax.vjp(proj.projection_simplex, x)[1](v)
      np.testing.assert_array_almost_equal(vjp, jnp.dot(v, jac_true))

    with self.subTest('Check Jacobian-vector product'):
      jvp = jax.jvp(proj.projection_simplex, (x,), (v,))[1]
      np.testing.assert_array_almost_equal(jvp, jnp.dot(jac_true, v))

  @parameterized.parameters(1.0, 0.8)
  def test_projection_simplex_vmap(self, scale):
    rng = np.random.RandomState(0)
    x = rng.randn(3, 50).astype(np.float32)
    scales = jnp.full(len(x), scale)

    p = jax.vmap(proj.projection_simplex)(x, scales)
    np.testing.assert_array_almost_equal(jnp.sum(p, axis=1), scales)
    np.testing.assert_array_equal(True, 0 <= p)
    np.testing.assert_array_equal(True, p <= scale)

  @parameterized.product(
      data_key=['array_1d', 'array_2d', 'tree'], scale=[1.0, 3.21]
  )
  def test_projection_l1_sphere(self, data_key, scale):
    x = self.data[data_key]
    p = proj.projection_l1_sphere(x, scale)
    np.testing.assert_almost_equal(optax.tree.norm(p, ord=1), scale, decimal=4)

  @parameterized.product(
      data_key=['array_1d', 'array_2d', 'tree'], scale=[1.0, 3.21]
  )
  def test_projection_l2_sphere(self, data_key, scale):
    x = self.data[data_key]
    p = proj.projection_l2_sphere(x, scale)
    np.testing.assert_almost_equal(optax.tree.norm(p), scale, decimal=4)

  @parameterized.product(
      data_key=['array_1d', 'array_2d', 'tree'],
      norm=['l1', 'l2', 'linf'],
  )
  def test_projection_ball(self, data_key, norm):
    """Check correctness of the projection onto a ball."""
    proj_fun, norm_fun = self.fns[norm]
    x = self.data[data_key]

    norm_value = norm_fun(x)

    with self.subTest('Check when input is already in the ball'):
      big_radius = norm_value * 2
      p = proj_fun(x, big_radius)
      np.testing.assert_array_almost_equal(x, p)

    with self.subTest('Check when input is on the boundary of the ball'):
      p = proj_fun(x, norm_value)
      np.testing.assert_array_almost_equal(x, p)

    with self.subTest('Check when input is outside the ball'):
      small_radius = norm_value / 2
      p = proj_fun(x, small_radius)
      np.testing.assert_almost_equal(norm_fun(p), small_radius, decimal=4)

  def test_projection_l2_ball_grad_at_zero(self):
    grad = jax.grad(proj.projection_l2_ball)(0.0)
    assert not jnp.isnan(grad)
    assert grad == 1.0

  def test_projection_l1_ball_grad_at_zero(self):
    grad = jax.grad(proj.projection_l1_ball)(0.0)
    assert not jnp.isnan(grad)
    assert grad == 1.0

  def test_projection_vector(self):
    x = (1.0, 2.0)
    a = (3.0, 4.0)
    y_actual = proj.projection_vector(x, a)
    y_expected = (33 / 25, 44 / 25)
    assert tree_allclose(y_actual, y_expected)

  def test_projection_hyperplane(self):
    x = (1.0, 2.0)
    a = (3.0, 4.0)
    b = 5.0
    y_actual = proj.projection_hyperplane(x, a, b)
    y_expected = (7 / 25, 26 / 25)
    print(y_actual)
    print(y_expected)
    assert tree_allclose(y_actual, y_expected)

  def test_projection_halfspace_1(self):
    x = (1.0, 2.0)
    a = (3.0, 4.0)
    b = 5.0
    y_actual = proj.projection_halfspace(x, a, b)
    y_expected = (7 / 25, 26 / 25)
    print(y_actual)
    print(y_expected)
    assert tree_allclose(y_actual, y_expected)

  def test_projection_halfspace_2(self):
    x = (1.0, -2.0)
    a = (3.0, 4.0)
    b = 5.0
    y_actual = proj.projection_halfspace(x, a, b)
    y_expected = x
    print(y_actual)
    print(y_expected)
    assert tree_allclose(y_actual, y_expected)


def tree_allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  # Replace this with optax.tree.allclose, once that's added.
  return all(jax.tree.leaves(jax.tree.map(lambda a, b: jnp.allclose(
      a, b, rtol=rtol, atol=atol, equal_nan=equal_nan), a, b)))


if __name__ == '__main__':
  absltest.main()
