"""Placing a user in an already-fitted projection.

Two placement strategies with genuinely different guarantees, which is the
whole reason both exist:

* PCA is linear, so a user vector maps into the displayed space *exactly*.
  That is testable as an identity, and it is tested as one here.
* UMAP has no transform for an unseen point, so the user is placed at a
  similarity-weighted average of nearby items. That is an approximation, and
  the only honest assertions are structural ones: it lands inside the hull of
  the items it averaged, and it moves toward whichever items the user actually
  resembles.
"""

from __future__ import annotations

import numpy as np
import pytest

from mercury_rec.retrieval.projection import barycentric_user_position, project_user_vector


@pytest.fixture
def basis() -> dict[str, np.ndarray | float]:
    """A 3-of-8 orthonormal PCA basis with a non-trivial centre and scale."""
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(8, 8))
    orthonormal, _ = np.linalg.qr(raw)
    return {
        "components": orthonormal[:3].astype(np.float32),
        "mean": rng.normal(size=8).astype(np.float32),
        "centre": np.array([0.4, -0.2, 0.1], dtype=np.float32),
        "radius": 2.5,
    }


class TestProjectUserVector:
    def test_reproduces_the_transform_the_items_went_through(
        self, basis: dict[str, np.ndarray | float]
    ) -> None:
        """The function must apply the projection AND the normalisation.

        Applying only the PCA basis puts the star in the right direction at
        the wrong scale - it still renders, still moves plausibly, and is
        wrong by a constant factor that nothing downstream can detect.
        """
        components = np.asarray(basis["components"])
        mean = np.asarray(basis["mean"])
        centre = np.asarray(basis["centre"])
        radius = float(basis["radius"])

        embedding = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        expected = ((embedding - mean) @ components.T - centre) / radius

        placed = project_user_vector(
            embedding, components=components, mean=mean, centre=centre, radius=radius
        )
        np.testing.assert_allclose(placed, expected, rtol=1e-6, atol=1e-6)

    def test_a_zero_radius_does_not_divide_by_zero(
        self, basis: dict[str, np.ndarray | float]
    ) -> None:
        """A degenerate projection - every item at one point - must not crash.

        It is reachable: a catalogue whose embeddings all collapse produces a
        zero radius, and returning inf would poison the scene rather than
        render a useless but finite one.
        """
        placed = project_user_vector(
            np.ones(8, dtype=np.float32),
            components=np.asarray(basis["components"]),
            mean=np.asarray(basis["mean"]),
            centre=np.asarray(basis["centre"]),
            radius=0.0,
        )
        assert np.all(np.isfinite(placed))

    def test_returns_float32(self, basis: dict[str, np.ndarray | float]) -> None:
        """The frontend reads these as a Float32Array; widening costs bytes."""
        placed = project_user_vector(
            np.ones(8, dtype=np.float32),
            components=np.asarray(basis["components"]),
            mean=np.asarray(basis["mean"]),
            centre=np.asarray(basis["centre"]),
            radius=float(basis["radius"]),
        )
        assert placed.dtype == np.float32


class TestBarycentricPosition:
    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

    def test_lands_inside_the_hull_of_the_items_it_averages(self) -> None:
        """A convex combination cannot escape the box its inputs occupy.

        This is the property that makes the placement safe to render: however
        unusual the user, the star stays inside the galaxy rather than flying
        off past the far clipping plane.
        """
        rng = np.random.default_rng(3)
        item_embeddings = self._normalise(rng.normal(size=(200, 16)).astype(np.float32))
        item_coords = rng.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32)
        user = self._normalise(rng.normal(size=16).astype(np.float32))

        position = barycentric_user_position(user, item_embeddings, item_coords, top_n=32)

        assert position.shape == (3,)
        assert np.all(position >= item_coords.min(axis=0) - 1e-5)
        assert np.all(position <= item_coords.max(axis=0) + 1e-5)

    def test_moves_toward_the_items_the_user_resembles(self) -> None:
        """The placement must track similarity, not just average everything.

        Two clusters, a user planted inside one of them: the star belongs with
        its own cluster. If it landed midway the view would be decorative -
        every user would sit near the centroid of the catalogue.
        """
        rng = np.random.default_rng(7)
        dim = 16

        left = self._normalise(
            np.tile(np.eye(dim, dtype=np.float32)[0], (60, 1))
            + rng.normal(scale=0.02, size=(60, dim)).astype(np.float32)
        )
        right = self._normalise(
            np.tile(np.eye(dim, dtype=np.float32)[1], (60, 1))
            + rng.normal(scale=0.02, size=(60, dim)).astype(np.float32)
        )
        item_embeddings = np.vstack([left, right])

        item_coords = np.vstack(
            [
                np.tile(np.array([-1.0, 0.0, 0.0], dtype=np.float32), (60, 1)),
                np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (60, 1)),
            ]
        )

        left_user = self._normalise(np.eye(dim, dtype=np.float32)[0])
        right_user = self._normalise(np.eye(dim, dtype=np.float32)[1])

        left_position = barycentric_user_position(left_user, item_embeddings, item_coords, top_n=32)
        right_position = barycentric_user_position(
            right_user, item_embeddings, item_coords, top_n=32
        )

        assert left_position[0] < -0.9
        assert right_position[0] > 0.9

    def test_top_n_larger_than_the_catalogue_is_not_an_error(self) -> None:
        """A demo preset can hold fewer items than the neighbour budget."""
        rng = np.random.default_rng(13)
        item_embeddings = self._normalise(rng.normal(size=(5, 8)).astype(np.float32))
        item_coords = rng.uniform(-1.0, 1.0, size=(5, 3)).astype(np.float32)
        user = self._normalise(rng.normal(size=8).astype(np.float32))

        position = barycentric_user_position(user, item_embeddings, item_coords, top_n=64)
        assert np.all(np.isfinite(position))
