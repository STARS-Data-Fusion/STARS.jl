# STARS.jl

Spatial Timeseries for Automated high-Resolution multi-Sensor data fusion (STARS) Julia Package

Margaret C. Johnson (she/her)<br>
[maggie.johnson@jpl.nasa.gov](mailto:maggie.johnson@jpl.nasa.gov)<br>
Principal investigator: lead of data fusion methodological development and Julia code implementations.<br>
NASA Jet Propulsion Laboratory 398L

Gregory H. Halverson (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer for data processing pipeline design and development, moving window implementation, and code organization and management.<br>
NASA Jet Propulsion Laboratory 329G

Jouni I. Susiluoto (he/him)<br>
[jouni.i.susiluoto@jpl.nasa.gov](mailto:jouni.i.susiluoto@jpl.nasa.gov)<br>
Technical contributor for methodology development, co- developer of Julia code for Kalman filtering recursion.
NASA Jet Propulsion Laboratory 398L

Kerry Cawse-Nicholson (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
Concept development and project management. Advised on technical and scientific requirements for application and mission integration.<br>
NASA Jet Propulsion Laboratory 329G

Joshua B. Fisher (he/him)<br>
[jbfisher@chapman.edu](mailto:jbfisher@chapman.edu)<br>
Concept development and project management<br>
Chapman University

Glynn C. Hulley (he/him)<br>
[glynn.hulley@jpl.nasa.gov](mailto:glynn.hulley@jpl.nasa.gov)<br>
Advised on technical and scientific requirements for application and mission integration.<br>
NASA Jet Propulsion Laboratory 329G

Nimrod Carmon (he/him)<br>
[nimrod.carmon@jpl.nasa.gov](mailto:nimrod.carmon@jpl.nasa.gov)<br>
Technical contributor for data processing, validation/verification, and hyperspectral resampling<br>
NASA Jet Propulsion Laboratory 398L

### Abstract

STARS is a general data fusion methodology utilizing spatiotemporal statistical models to optimally combine high spatial resolution VSWIR measurements with high temporal resolution measurements from multiple instruments. The methods are highly-scalable, able to fuse <100 m spatial resolution products in near-real time (<24 hrs) on regional to global scales, to facilitate online data processing as well as large-scale reprocessing of mission datasets. The statistical spatiotemporal modeling framework provides with each fused surface reflectance product associated pixel-level uncertainties incorporating any known data source measurement uncertainties, bias characteristics, and degree of historical data missingness. 

The specific capabilities offered by STARS are: 
1. automatic, high-resolution spatial and temporal gap-filling, 
2. a tunable fusion framework allowing the user to choose a level of accuracy vs computational complexity, and 
3. quantifiable uncertainties that can be used for downstream product sensitivity/uncertainty assessments and that can be incorporated into higher-order data product quality flags. 

STARS is a significant advancement for surface reflectance data fusion and for quantifying (and potentially reducing) the uncertainty associated with satellite-derived inputs in retrievals of science quantities of interest.


### This software accomplishes the following:

Implements a novel, statistical data fusion methodology called STARS (Spatial Timeseries for Automated high- Resolution multi-Sensor data fusion) to provide daily, high spatial resolution estimates of remotely-sensed imagery by combining data from multiple spectrometers (e.g. Landsat/Sentinel/MODIS/VIIRS).

### What are the unique features of the software?

Provides mission scalable, automated, spatiotemporal gap-filling to produce high resolution imagery with appropriately quantified uncertainties. No current data fusion methodology/software can provide such capabilities.

### What improvements have been made over existing similar software application?

Current state-of-the-art data fusion software packages (e.g. Python implementations of STARFM and its extensions) are not computational scalable nor do they provide a means for uncertainty quantification. STARS provides predictions of satellite imagery at resolutions not achievable by single instruments with associated quantified uncertainties. The implementation is embarrassingly parallel which facilitates global scale processing.

### What problems are you trying to solve in the software?

Due to physical limitations of individual sensors, multispectral imaging spectroscopy instruments can achieve either high ($<$ 100 m) spatial resolution, or high (daily) temporal revisit, but not both. Statistical data fusion software leveraging multiple spectroscopy instruments with different resolutions provides a means to produce high spatial and temporal resolution data products.

### Does your work relate to current or future NASA (include reimbursable) work that has value to the conduct of aeronautical and space activities? If so, please explain:

Yes, the work is directly relevant to NASA Earth science missions, particularly ECOSTRESS and SBG.
The software provides the capability of producing high resolution visibile to shortwave infrared (VSWIR) images coincident to TIR instrument overpasses. This greatly improves ECOSTRESS's, and potentially SBG's, ability to estimate evapotranspiration. This capability potentially benefits all thermal remote sensing missions and evapotranspiration remote sensing projects. Additionally, STARS allows for increased usability of NASA satellite products by sharpening imagery to resolutions enabling ecosystem studies of phenology and carbon cycling with links to biodiversity, snow, and volcanic responses.


### What advantages does this software have over existing software?

The STARS algorithm is computationally scalable and provides uncertainty quantification. Existing competitors, mainly the STARFM algorithm, are not scalable and do not provide uncertainties.

### Are there any known commercial applications? What are they? What else is currently on the market that is similar?

This software is intended for science data system use and scientific research. The closest publicly available equivalent is the STARFM algorithm.


