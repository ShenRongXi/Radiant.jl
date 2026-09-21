# Shared threading constants used across particle_transport/ source files.
# Centralized here so that every file can rely on the same definitions without
# duplicating `const` declarations (which Julia 1.9+ disallows even for same-value
# redefinitions).

# Legacy ix-only parallel threshold: minimum number of voxels along the x-axis
# before `@threads :static for ix` is worth its overhead.  For 1D problems this
# avoids parallelizing a trivial number of voxels; for 2D/3D the total-voxel
# threshold below is the controlling guard.
const PAR_MIN_NX = 80

# Total-voxel threshold for the new voxel-parallel (CartesianIndices) strategy.
# The guard `Nxyz >= PAR_MIN_NVOXELS` gates the ix·iy·iz parallel branch.  A value
# of 500 keeps the per-thread chunk large enough that barrier overhead does not
# dominate on small 1D grids.
const PAR_MIN_NVOXELS = 500
