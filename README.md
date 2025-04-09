# BridgeRiskEQ
This is a quick and easy example of running regional seismic risk analysis of bridges in Memphis, Tennessee. The bridge database (i.e., locations and other key attributes) is collected from the National bridge Inventory Database, an open-access repository of bridges in the US that can be accesed in this link: https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm

The damage to bridges is modeled using a fragility model, which takes the local hazard intensity as input and provides the probability of exceeding various prescribed damage levels (Slight, Moderate, Extensive, and Complete) as output. The fragility models are based on the models available in HAZUS (https://www.fema.gov/sites/default/files/2020-10/fema_hazus_earthquake_technical_manual_4-2.pdf
).

The local hazard intensity at the location of each bridge is based on a raster layer of seismic hazard maps developed by USGS: https://www.usgs.gov/programs/earthquake-hazards/science/2014-united-states-lower-48-seismic-hazard-long-term-model
.

Since each of the datasets above are obtained at various spatial resolutions (e.g. national, state), a set of regional boundary shapefiles are used to clip the datasets within the study area, which have been downloaded from https://catalog.data.gov/dataset/tiger-line-shapefile-2019-county-shelby-county-tn-all-roads-county-based-shapefile
.

The transportation network data is downloaded from OpenStreetMap: https://www.openstreetmap.org/#map=4/37.65/-82.00
. The OSM data has been downloaded, preprocessed and saved in the form of shapefiles in this directory.

 
