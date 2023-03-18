from itertools import product
from typing import Mapping, Any, Sequence, Optional, Union, List
from clearml.automation.parameters import Parameter

import clearml.automation.parameters as parameters

class UniformRange(Parameter):
    """
    Uniform randomly sampled hyper-parameter object.
    """

    def __init__(
            self,
            min_value,
            max_value,
            step_size=None,
            include_max_value=True
    ):
        # type: (float, float, Optional[float], bool) -> ()
        """
        Create a parameter to be sampled by the SearchStrategy
        :param float min_value: The minimum sample to use for uniform random sampling.
        :param float max_value: The maximum sample to use for uniform random sampling.
        :param float step_size: If not ``None``, set step size (quantization) for value sampling.
        :param bool include_max_value: Range includes the ``max_value``
            The values are:
            - ``True`` - The range includes the ``max_value`` (Default)
            - ``False`` -  Does not include.
        """
        super(UniformRange, self).__init__(name=None)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.step_size = float(step_size) if step_size is not None else None
        self.include_max = include_max_value

    def get_value(self):
        # type: () -> float
        """
        Return uniformly sampled value based on object sampling definitions.
        :return: random value [self.min_value, self.max_value)
        """
        if not self.step_size:
            return self._random.uniform(self.min_value, self.max_value)
        steps = (self.max_value - self.min_value) / self.step_size
        return self.min_value + (self._random.randrange(start=0, stop=round(steps)) * self.step_size)

    def to_list(self):
        # type: () -> Sequence[float]
        """
        Return a list of all the valid values of the Parameter. If ``self.step_size`` is not defined, return 100 points
        between min/max values.
        :return: list of float
        """
        step_size = self.step_size or (self.max_value - self.min_value) / 100.
        steps = (self.max_value - self.min_value) / step_size
        values = [self.min_value + v*step_size for v in range(0, int(steps))]
        if self.include_max and (not values or values[-1] < self.max_value):
            values.append(self.max_value)
        return values

class UniformIntegerRange(Parameter):
    """
    Uniform randomly sampled integer Hyper-Parameter object.
    """

    def __init__(
            self, 
            min_value, 
            max_value, 
            step_size=1, 
            include_max_value=True
    ):
        # type: (int, int, int, bool) -> ()
        """
        Create a parameter to be sampled by the SearchStrategy.
        :param int min_value: The minimum sample to use for uniform random sampling.
        :param int max_value: The maximum sample to use for uniform random sampling.
        :param int step_size: The default step size is ``1``.
        :param bool include_max_value: Range includes the ``max_value``
            The values are:
            - ``True`` - Includes the ``max_value`` (Default)
            - ``False`` - Does not include.
        """
        super(UniformIntegerRange, self).__init__(name=None)
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        self.step_size = int(step_size) if step_size is not None else None
        self.include_max = include_max_value

    def get_value(self):
        # type: () -> int
        """
        Return uniformly sampled value based on object sampling definitions.
        :return: random value [self.min_value, self.max_value)
        """
        return self._random.randrange(
            start=self.min_value, step=self.step_size,
            stop=self.max_value + (0 if not self.include_max else self.step_size))

    def to_list(self):
        # type: () -> Sequence[int]
        """
        Return a list of all the valid values of the Parameter. If ``self.step_size`` is not defined, return 100 points
        between minmax values.
        :return: list of int
        """
        values = list(range(self.min_value, self.max_value, self.step_size))
        if self.include_max and (not values or values[-1] < self.max_value):
            values.append(self.max_value)
        return values

class NargsParameterSet(Parameter):
    def __init__(self, name, parameter_combinations=[]):
        # type: (str, List[Union[UniformRange, UniformIntegerRange]]) -> ()
        """
        Uniformly sample values form a list of discrete options (combinations) of parameters.
        :param list parameter_combinations: The list/tuple of valid parameter combinations.
        """
        super(NargsParameterSet, self).__init__(name=name)
        self.values = []
        for i, p in enumerate(parameter_combinations):
            locals()[str(i)] = p
            self.values.append(str(i))
        self.__dict__.update(locals())
        del self.__dict__['self']
        del self.__dict__['__class__']
        del self.__dict__['i']
        del self.__dict__['p']
        del self.__dict__['parameter_combinations']

    def get_value(self):
        # type: () -> Mapping[str, List[Any]]
        """
        Return uniformly sampled value from the valid list of values.
        :return: {self.name: random entry from self.value}
        """
        return {self.name: [getattr(self, v).get_value() for v in self.values]}

    def to_list(self):
        # type: () -> Sequence[Mapping[str, Any]]
        """
        Return a list of all the valid values of the Parameter.
        :return: list of dicts {name: value}
        """
        combinations = list(product(*[getattr(self, v).to_list() for v in self.values]))
        return [{self.name: c} for c in combinations]

    def to_dict(self):
        # type: () -> Mapping[str, Union[str, Parameter]]
        """
        Return a dict representation of the Parameter object. Used for serialization of the Parameter object.
        :return:  dict representation of the object (serialization).
        """
        serialize = {self._class_type_serialize_name: str(self.__class__).split('.')[-1][:-2]}
        # noinspection PyCallingNonCallable
        serialize.update(dict(((k, v.to_dict() if hasattr(v, 'to_dict') else v) for k, v in self.__dict__.items())))
        return serialize
setattr(parameters, NargsParameterSet.__name__, NargsParameterSet)
setattr(parameters, UniformIntegerRange.__name__, UniformIntegerRange)
setattr(parameters, UniformRange.__name__, UniformRange)

if __name__=="__main__":
    nargs = NargsParameterSet('Args/smooth', parameter_combinations=[
                UniformRange(min_value=0.0, max_value=1, step_size=0.5, include_max_value=False),
                UniformRange(min_value=1, max_value=2, step_size=0.5, include_max_value=False),
                UniformRange(min_value=2, max_value=3, step_size=0.5, include_max_value=False),
                UniformRange(min_value=3, max_value=4, step_size=0.5, include_max_value=False),
            ])
    print(dict(nargs.__dict__.items()))
    print(nargs.to_list())
    print(nargs.get_value())
    print(nargs.to_dict())
    print(Parameter.from_dict(nargs.to_dict()).values)
    print(Parameter.from_dict(nargs.to_dict()).__dict__['0'].name)
    print(Parameter.from_dict(nargs.to_dict()).__dict__['0'].min_value)
