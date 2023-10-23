from pd.data_lab.config import distribution as pd_distribution

from paralleldomain.utilities import inherit_docs


@inherit_docs(level=2)
class NormalDistribution(pd_distribution.NormalDistribution):
    ...


@inherit_docs(level=2)
class ContinousUniformDistribution(pd_distribution.ContinousUniformDistribution):
    ...


@inherit_docs(level=2)
class Bucket(pd_distribution.Bucket):
    ...


@inherit_docs(level=2)
class CategoricalDistribution(pd_distribution.CategoricalDistribution):
    ...


@inherit_docs(level=2)
class ConstantDistribution(pd_distribution.ConstantDistribution):
    ...


@inherit_docs(level=2)
class Distribution(pd_distribution.Distribution):
    ...
