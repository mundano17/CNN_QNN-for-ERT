import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert
import pandas as pd
import matplotlib.pyplot as plt

# Create a more realistic world model
world = mt.createWorld(start=[-20, 0], end=[20, -15],
                       layers=[-2, -5, -10], worldMarker=1)

# Create anomalies with different resistivity values
anomalies = [
    mt.createCircle(pos=[-6, -3.5], radius=1.5, marker=2),
    mt.createCircle(pos=[4, -3], radius=1.8, marker=3),
    mt.createCircle(pos=[-2, -7], radius=2, marker=4),
    mt.createPolygon([[-10, -8], [-8, -10], [-12, -10]], marker=5)
]

# Append anomalies to world
for anomaly in anomalies:
    world += anomaly

# Create mesh
mesh = mt.createMesh(world, quality=33.5, area=0.25)

# Set resistivity values
resistivities = {
    0: 100,  # default/unmarked cells
    1: 100,  # background soil
    2: 500,  # high resistivity anomaly
    3: 250,  # medium resistivity anomaly
    4: 20,   # low resistivity anomaly (clay)
    5: 350   # bedrock
}

# Create resistivity array for the entire mesh
rhomap = pg.solver.parseMapToCellArray(resistivities, mesh)

# Visualize the true model
fig, ax = plt.subplots(figsize=(10, 6))
pg.show(mesh, data=rhomap, label='Resistivity [Ωm]',
        cMap='Spectral_r', logScale=True, ax=ax)
plt.title('True Resistivity Model')
plt.savefig('true_resistivity_model.png', dpi=300)
plt.close()

# Create electrode array
n_electrodes = 41
electrodes = np.linspace(-15, 15, n_electrodes)

# Create just a dipole-dipole array (more stable)
scheme = ert.createERTData(electrodes, schemeName='dd')
print(f"Total measurements: {scheme.size()}")

# Simulate ERT data
ert_manager = ert.ERTManager()
data = ert_manager.simulate(
    mesh=mesh,
    res=rhomap,
    scheme=scheme,
    noiseLevel=2,  # 2% relative noise
    noiseAbs=1e-5  # Absolute noise floor
)

# Calculate geometric factors and error estimates
data['k'] = ert.geometricFactors(data)
data['err'] = ert.estimateError(data, absoluteError=0.001, relativeError=0.02)

# Filter out problematic measurements
data_filtered = data.copy()
data_filtered.markInvalid(data_filtered['rhoa'] < 0)
data_filtered.markInvalid(data_filtered['rhoa'] > 1000)
data_filtered.removeInvalid()

print(f"Measurements after filtering: {data_filtered.size()}")

# Extract electrode positions and measurements
a = np.array(data_filtered['a'])
b = np.array(data_filtered['b'])
m = np.array(data_filtered['m'])
n = np.array(data_filtered['n'])
rhoa = np.array(data_filtered['rhoa'])
k = np.array(data_filtered['k'])
err = np.array(data_filtered['err'])

# Calculate useful features for ML
midpoint_ab = (electrodes[a] + electrodes[b]) / 2
midpoint_mn = (electrodes[m] + electrodes[n]) / 2
spacing_ab = np.abs(electrodes[a] - electrodes[b])
spacing_mn = np.abs(electrodes[m] - electrodes[n])
array_length = np.maximum(electrodes[b], electrodes[n]) - np.minimum(electrodes[a], electrodes[m])
dipole_separation = np.abs((electrodes[a] + electrodes[b])/2 - (electrodes[m] + electrodes[n])/2)
pseudo_depth = dipole_separation / 2  # Approximate investigation depth

# Create dataset with features
df = pd.DataFrame({
    'a_pos': electrodes[a],
    'b_pos': electrodes[b],
    'm_pos': electrodes[m],
    'n_pos': electrodes[n],
    'midpoint_ab': midpoint_ab,
    'midpoint_mn': midpoint_mn,
    'spacing_ab': spacing_ab,
    'spacing_mn': spacing_mn,
    'array_length': array_length,
    'dipole_separation': dipole_separation,
    'pseudo_depth': pseudo_depth,
    'k_factor': k,
    'rhoa': rhoa  # Target variable
})

# Save the dataset
df.to_csv('synthetic_ert_multi_anomalies.csv', index=False)
print("Enhanced data saved to synthetic_ert_multi_anomalies.csv")

# Show resistivity distribution
plt.figure(figsize=(10, 6))
plt.hist(df['rhoa'], bins=40, alpha=0.7)
plt.xlabel('Apparent Resistivity (Ωm)')
plt.ylabel('Frequency')
plt.title('Distribution of Apparent Resistivity Values')
plt.grid(True, alpha=0.3)
plt.savefig('resistivity_distribution.png')
plt.close()

print("Data generation complete!")
