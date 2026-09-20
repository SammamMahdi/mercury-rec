"use client";

/**
 * The Recommendation Galaxy.
 *
 * Real two-tower item embeddings, projected to 3D offline by UMAP or PCA and
 * served as binary buffers. Distance in the scene is embedding similarity, so
 * the clusters are the model's own structure rather than a layout algorithm
 * applied to arbitrary points.
 *
 * Three decisions carry the performance of this page:
 *
 * - **One `<points>` object** holds all 6,000 items in a single
 *   BufferGeometry. Rendering 6,000 `<mesh>` components - the canonical React
 *   Three Fiber mistake - puts 6,000 objects in the scene graph and in
 *   React's tree, and drops the frame rate into single digits before anything
 *   moves.
 * - **View state lives in a typed array**, not in React. Switching to
 *   "candidates only" writes a Float32Array and flags it dirty; it re-renders
 *   no component and touches no scene graph.
 * - **`frameloop="demand"`** with auto-rotate off means the GPU does nothing
 *   at all while someone reads the side panel. With auto-rotate on the scene
 *   animates continuously and this saves nothing, which is exactly why the
 *   rotation is a control rather than a default.
 *
 * Hover is handled through React Three Fiber's own pointer events rather than
 * a raycast inside `useFrame`. Under an on-demand frame loop a `useFrame`
 * raycast only runs when something else already caused a render, so moving
 * the pointer across a still scene would highlight nothing.
 */

import { OrbitControls } from "@react-three/drei";
import { Canvas, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";

import { VERTICALS } from "@/lib/constants";

export interface GalaxyPoints {
  x: Float32Array;
  y: Float32Array;
  z: Float32Array;
  itemId: Int32Array;
  vertical: Int32Array;
  price: Float32Array;
}

export type GalaxyView = "all" | "candidates" | "ranked" | "final";

/** Per-point display tier, packed into a vertex attribute. */
const TIER_DIM = 0;
const TIER_NORMAL = 1;
const TIER_CANDIDATE = 2;
const TIER_RANKED = 3;
const TIER_FINAL = 4;

/** sRGB, 0-1 per channel. Matches what the fragment shader writes. */
type Rgb = readonly [number, number, number];

const FALLBACK_COLOUR: Rgb = [0.49, 0.83, 0.99];

/**
 * Resolve the vertical palette from CSS to RGB, once per page load.
 *
 * The palette is authored in `oklch()`, which three.js cannot parse - handing
 * it to `THREE.Color` throws, and a catch-all fallback would quietly paint all
 * six verticals the same colour while the legend claimed otherwise. So the
 * conversion is delegated to the only colour engine guaranteed to agree with
 * the rest of the page: the browser's own, through a 1x1 canvas.
 *
 * The values stay in sRGB and are written straight to `gl_FragColor`. A raw
 * ShaderMaterial gets none of three.js's colour-space conversion chunks, so
 * sRGB in and sRGB out is the combination that renders the colour the
 * stylesheet actually specified.
 *
 * Cached at module scope rather than held in state: this component only ever
 * renders in the browser, the answer cannot change between mounts, and a
 * state update purely to deliver a constant would cost a second render of the
 * entire scene.
 */
let cachedPalette: Rgb[] | null = null;

function verticalPalette(): Rgb[] {
  if (cachedPalette) return cachedPalette;
  if (typeof document === "undefined") return VERTICALS.map(() => FALLBACK_COLOUR);

  const styles = getComputedStyle(document.documentElement);
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 1;
  const context = canvas.getContext("2d", { willReadFrequently: true });

  cachedPalette = VERTICALS.map((vertical) => {
    const property = vertical.color.replace(/^var\(|\)$/g, "");
    const raw = styles.getPropertyValue(property).trim();
    if (!context || !raw) return FALLBACK_COLOUR;

    // An unparseable value leaves fillStyle at the previous one, so seeding
    // it with black makes that case detectable rather than silent.
    context.fillStyle = "#000000";
    context.fillStyle = raw;
    context.fillRect(0, 0, 1, 1);

    const [r, g, b] = context.getImageData(0, 0, 1, 1).data;
    if (r === 0 && g === 0 && b === 0) return FALLBACK_COLOUR;
    return [r / 255, g / 255, b / 255] as const;
  });
  return cachedPalette;
}

/* -------------------------------------------------------------------------- */

function PointCloud({
  points,
  candidates,
  ranked,
  finals,
  view,
  onHover,
  onSelect,
}: {
  points: GalaxyPoints;
  candidates: Set<number>;
  ranked: Set<number>;
  finals: Set<number>;
  view: GalaxyView;
  onHover: (index: number | null) => void;
  onSelect: (index: number) => void;
}) {
  const colours = verticalPalette();
  const count = points.itemId.length;

  /* Positions and colours are fixed for the life of a projection, so they are
     built once. Only the tier attribute changes afterwards. */
  const { positions, colourAttribute } = useMemo(() => {
    const positionArray = new Float32Array(count * 3);
    const colourArray = new Float32Array(count * 3);

    for (let i = 0; i < count; i += 1) {
      positionArray[i * 3] = points.x[i];
      positionArray[i * 3 + 1] = points.y[i];
      positionArray[i * 3 + 2] = points.z[i];

      const [r, g, b] = colours[points.vertical[i]] ?? colours[0];
      colourArray[i * 3] = r;
      colourArray[i * 3 + 1] = g;
      colourArray[i * 3 + 2] = b;
    }
    return { positions: positionArray, colourAttribute: colourArray };
  }, [points, colours, count]);

  /* Derived, not mutated in place. A fresh Float32Array gives React Three
     Fiber a changed `args`, so it rebuilds the attribute and re-uploads it;
     24KB on a toggle a visitor performs a handful of times is not worth the
     mutable-buffer machinery it would take to avoid. */
  const tiers = useMemo(() => {
    const array = new Float32Array(count);
    const hasSelection = candidates.size > 0 || ranked.size > 0 || finals.size > 0;

    for (let i = 0; i < count; i += 1) {
      const id = points.itemId[i];
      let tier = hasSelection ? TIER_DIM : TIER_NORMAL;

      if (finals.has(id)) tier = TIER_FINAL;
      else if (ranked.has(id)) tier = TIER_RANKED;
      else if (candidates.has(id)) tier = TIER_CANDIDATE;

      // The view filter pushes anything below the chosen stage back to dim,
      // which is how "show me only what survived ranking" reads on screen.
      if (view === "candidates" && tier < TIER_CANDIDATE) tier = TIER_DIM;
      else if (view === "ranked" && tier < TIER_RANKED) tier = TIER_DIM;
      else if (view === "final" && tier < TIER_FINAL) tier = TIER_DIM;

      array[i] = tier;
    }
    return array;
  }, [candidates, ranked, finals, view, points.itemId, count]);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        transparent: true,
        // Additive blending without depth writes: sorting thousands of
        // translucent sprites back-to-front would cost more than the scene is
        // worth, and glow reads correctly without the sort.
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        vertexShader: /* glsl */ `
          attribute vec3 aColour;
          attribute float aTier;
          varying vec3 vColour;
          varying float vTier;

          void main() {
            vColour = aColour;
            vTier = aTier;

            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

            // Size by tier, so the selection survives colour-vision
            // deficiency and greyscale screenshots instead of relying on hue.
            float base = aTier < 0.5 ? 2.0
                       : aTier < 1.5 ? 3.5
                       : aTier < 2.5 ? 6.0
                       : aTier < 3.5 ? 9.0
                       : 14.0;

            gl_PointSize = base * (300.0 / -mvPosition.z);
            gl_Position = projectionMatrix * mvPosition;
          }
        `,
        fragmentShader: /* glsl */ `
          varying vec3 vColour;
          varying float vTier;

          void main() {
            // Round, soft-edged sprites. The default square quad makes a
            // point cloud read as compression noise.
            float distance = length(gl_PointCoord - vec2(0.5));
            if (distance > 0.5) discard;

            float falloff = smoothstep(0.5, 0.05, distance);
            float alpha = vTier < 0.5 ? 0.09 : vTier < 1.5 ? 0.45 : 0.95;

            gl_FragColor = vec4(vColour, alpha * falloff);
          }
        `,
      }),
    [],
  );

  // Dispose explicitly: a ShaderMaterial holds a compiled GPU program that
  // React unmounting the component does not release.
  useEffect(() => () => material.dispose(), [material]);

  const handleMove = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    onHover(event.index ?? null);
  };

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    if (event.index !== undefined) onSelect(event.index);
  };

  return (
    <points
      material={material}
      onPointerMove={handleMove}
      onPointerOut={() => onHover(null)}
      onClick={handleClick}
    >
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-aColour" args={[colourAttribute, 3]} />
        <bufferAttribute attach="attributes-aTier" args={[tiers, 1]} />
      </bufferGeometry>
    </points>
  );
}

/* -------------------------------------------------------------------------- */

/**
 * The user's own position in the embedding space.
 *
 * Deliberately a different shape from every item - a solid core inside a
 * wireframe shell - rather than a brighter dot, because in PCA mode this is an
 * exact projection and in UMAP mode it is an approximation, and neither should
 * be mistaken for one more catalogue item.
 */
function UserStar({ position }: { position: [number, number, number] }) {
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[0.022, 16, 16]} />
        <meshBasicMaterial color="#ffffff" toneMapped={false} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.055, 16, 16]} />
        <meshBasicMaterial
          color="#ffffff"
          wireframe
          transparent
          opacity={0.32}
          toneMapped={false}
        />
      </mesh>
    </group>
  );
}

/* -------------------------------------------------------------------------- */

export function GalaxyCanvas({
  points,
  candidates,
  ranked,
  finals,
  view,
  userPosition,
  autoRotate,
  onHoverItem,
  onSelectItem,
}: {
  points: GalaxyPoints;
  candidates: Set<number>;
  ranked: Set<number>;
  finals: Set<number>;
  view: GalaxyView;
  userPosition: [number, number, number] | null;
  autoRotate: boolean;
  onHoverItem: (itemId: number | null) => void;
  onSelectItem: (itemId: number) => void;
}) {
  return (
    <Canvas
      // Capped device pixel ratio. At native ratio on a high-density display
      // this shades four times the fragments for a point cloud that gains
      // almost nothing from the extra samples.
      dpr={[1, 1.75]}
      frameloop="demand"
      camera={{ position: [0, 0, 2.6], fov: 50, near: 0.01, far: 100 }}
      // Three.js defaults the point threshold to one world unit. In a scene
      // normalised to a unit sphere that is the radius of the whole galaxy, so
      // one pointer move would intersect most of the catalogue. The other
      // entries are the library defaults, restated because the type requires
      // the whole parameter set.
      raycaster={{
        params: {
          Mesh: {},
          Line: { threshold: 1 },
          LOD: {},
          Points: { threshold: 0.022 },
          Sprite: {},
        },
      }}
      gl={{ antialias: false, powerPreference: "high-performance", alpha: true }}
      style={{ background: "transparent" }}
    >
      <PointCloud
        points={points}
        candidates={candidates}
        ranked={ranked}
        finals={finals}
        view={view}
        onHover={(index) => onHoverItem(index === null ? null : points.itemId[index])}
        onSelect={(index) => onSelectItem(points.itemId[index])}
      />
      {userPosition ? <UserStar position={userPosition} /> : null}
      <OrbitControls
        enablePan={false}
        enableDamping
        dampingFactor={0.08}
        minDistance={0.6}
        maxDistance={6}
        autoRotate={autoRotate}
        autoRotateSpeed={0.35}
      />
    </Canvas>
  );
}
