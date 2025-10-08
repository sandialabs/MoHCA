# MoHCA

MoHCA is software for performing MOdel-free Hosting Capacity Analysis for electric distribution networks. It utilizes AMI data to calculate hosting capacity without the need for a circuit model. This approach has significant speed and accuracy advantages over traditional circuit-based hosting capacity analysis.

### Installation

With Python 3 installed: `pip install git+https://github.com/sandialabs/MoHCA/`


### Usage

On the command line:

`mohca_cl algo_name in_file.csv out_file_name.csv`

Where `algo_name` is one of {sandia1, iastate}. 

### Input/Output Data Formats

Please see the [OMF wiki entry on hostingCapacity](https://github.com/dpinney/omf/wiki/Models-~-hostingCapacity) for details on the input and output data formats, as well as some background on the methodology.