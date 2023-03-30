# PD SDK Architecture

## Model Classes

![image info](../images/pd-sdk-uml.png)

PD SDK **Datasets** contain references to **Scenes**. **Scenes** are either unordered (**UnorderedScenes**) or ordered (**Scene**).
**UnorderedScenes** are just collections of single frame data, whereas Scenes have frames that are temporally ordered.
Each UnorderedScene/Scene consists of a several **Frames**. Each **Frame** has a potentially changing amount of sensors that have data for that frame.
The data of a specific sensor on a specific frame is stored in a **SensorFrame**.
To support this the model is designed to represent an arbitrary sensor rig that collects sequential or non-sequential
data that may be annotated with different annotations.

## Decoders
In order to support different data formats PD SDK uses dataset format specific Decoders that are tasked with converting
the respective dataset format into the PD SDK common Python objects (aka model classes). The below example shows the
relationship between the model classes and the decoders. Each model class has a reference to a decoder reponsible
for laoding certain values for that class. For example the sensors avaialable in a Frame class, or the annotations
available in a SensorFrame.
Each Decoding Format then has to define how to load those values from the respective file storage format.
So there will we a subclass of each decoder for each file format.

If a format has no Radar, Lidar or Image data, the implementation of those decoders can simply be skipped,
since a Frame always lists its available Sensors which depend on the stored data.


![image info](../images/pd-sdk-uml-decoder.png)


PD SDK follows the principal of lazy loading data, meaning that any data is loaded as late as possible to ensure quick
browsing through datasets. Furthermore PD SDK contains an encoding module tasked with saving model classes into specific dataset formats.
This can be useful if you have an existing data pipeline that works on a certain format and you want to convert a
dataset to this format to be compatible with your infrastructure.


## Encoders

Documentation coming soon!
