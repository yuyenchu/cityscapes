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

from packaging import version
import clearml
if version.parse(clearml.__version__)>version.parse('1.9.3'):
    print('Detect clearml version > 1.9.3, activate patch for clearml.automation.HyperParameterOptimizer._report_completed_status')
    import typing
    from copy import copy, deepcopy
    from clearml import Task
    def patchFunc(self, completed_jobs, cur_completed_jobs, task_logger, title, force=False):
        ### replace call of __sort_jobs_by_objective since it's not accessible
        if not completed_jobs:
            return []
        job_ids_sorted_by_objective = list(sorted(
            completed_jobs.keys(), key=lambda x: completed_jobs[x][0], reverse=bool(self.objective_metric.sign >= 0)))
        ###
        best_experiment = \
            (self.objective_metric.get_normalized_objective(job_ids_sorted_by_objective[0]),
             job_ids_sorted_by_objective[0]) \
            if job_ids_sorted_by_objective else (float('-inf'), None)
        if force or cur_completed_jobs != set(completed_jobs.keys()):
            pairs = []
            labels = []
            created_jobs = copy(self.optimizer.get_created_jobs_ids())
            created_jobs_tasks = self.optimizer.get_created_jobs_tasks()
            id_status = {j_id: j_run.status() for j_id, j_run in created_jobs_tasks.items()}
            for i, (job_id, params) in enumerate(created_jobs.items()):
                value = self.objective_metric.get_objective(job_id)
                if job_id in completed_jobs:
                    if value != completed_jobs[job_id][0]:
                        iteration_value = self.objective_metric.get_current_raw_objective(job_id)
                        completed_jobs[job_id] = (
                            value,
                            iteration_value[0] if iteration_value else -1,
                            copy(dict(status=id_status.get(job_id), **params)))
                    elif completed_jobs.get(job_id):
                        completed_jobs[job_id] = (completed_jobs[job_id][0],
                                                  completed_jobs[job_id][1],
                                                  copy(dict(status=id_status.get(job_id), **params)))
                    pairs.append((i, completed_jobs[job_id][0]))
                    labels.append(str(completed_jobs[job_id][2])[1:-1])
                elif value is not None:
                    pairs.append((i, value))
                    labels.append(str(params)[1:-1])
                    iteration_value = self.objective_metric.get_current_raw_objective(job_id)
                    completed_jobs[job_id] = (
                        value,
                        iteration_value[0] if iteration_value else -1,
                        copy(dict(status=id_status.get(job_id), **params)))
                    # callback new experiment completed
                    if self._experiment_completed_cb:
                        normalized_value = self.objective_metric.get_normalized_objective(job_id)
                        if normalized_value is not None and normalized_value > best_experiment[0]:
                            best_experiment = normalized_value, job_id
                        c = completed_jobs[job_id]
                        print('Calling user callback')
                        self._experiment_completed_cb(job_id, c[0], c[1], c[2], best_experiment[1])

            if pairs:
                print('Updating job performance summary plot/table')
                task_logger.report_scatter2d(
                    title='Optimization Objective', series=title,
                    scatter=pairs, iteration=0, labels=labels,
                    mode='markers', xaxis='job #', yaxis='objective')
            job_ids = list(completed_jobs.keys())
            job_ids_sorted_by_objective = sorted(
                job_ids, key=lambda x: completed_jobs[x][0], reverse=bool(self.objective_metric.sign >= 0))
            columns = list(sorted(set([c for k, v in completed_jobs.items() for c in v[2].keys()])))
            table_values = [['task id', 'objective', 'iteration'] + columns]
            table_values += \
                [([job, completed_jobs[job][0], completed_jobs[job][1]] +
                  [completed_jobs[job][2].get(c, '') for c in columns]) for job in job_ids_sorted_by_objective]
            task_link_template = self._task.get_output_log_web_page() \
                .replace('/{}/'.format(self._task.project), '/{project}/') \
                .replace('/{}/'.format(self._task.id), '/{task}/')
            table_values_with_links = deepcopy(table_values)
            for i in range(1, len(table_values_with_links)):
                task_id = table_values_with_links[i][0]
                project_id = created_jobs_tasks[task_id].task.project \
                    if task_id in created_jobs_tasks else '*'
                table_values_with_links[i][0] = '<a href="{}"> {} </a>'.format(
                    task_link_template.format(project=project_id, task=task_id), task_id)
            task_logger.report_table(
                "summary", "job", 0, table_plot=table_values_with_links,
                extra_layout={"title": "objective: {}".format(title)})
            if len(table_values) > 1:
                table_values_columns = [[row[i] for row in table_values] for i in range(len(table_values[0]))]
                table_values_columns = \
                    [[table_values_columns[0][0]] + [c[:6]+'...' for c in table_values_columns[0][1:]]] + \
                    table_values_columns[2:-1] + [[title]+table_values_columns[1][1:]]
                pcc_dims = []
                for col in table_values_columns:
                    try:
                        values = [float(v) for v in col[1:]]
                        d = dict(label=col[0], values=values)
                    except (ValueError, TypeError):
                        values = list(range(len(col[1:])))
                        ticks = col[1:]
                        ### patch for unhashable objects
                        unique_ticks = list(set([t if isinstance(t, typing.Hashable) else tuple(t) for t in ticks]))
                        ###
                        d = dict(label=col[0], values=values, tickvals=values, ticktext=ticks)
                        if len(ticks) != len(unique_ticks): 
                            ticktext = {key: i for i, key in enumerate(unique_ticks)}
                            d["values"] = [ticktext[tick] for tick in ticks]
                            d["tickvals"] = list(range(len(ticktext)))
                            d["ticktext"] = list(sorted(ticktext, key=ticktext.get))
                    pcc_dims.append(d)
                plotly_pcc = dict(
                    data=[dict(
                        type='parcoords',
                        line=dict(colorscale='Viridis',
                                  reversescale=bool(self.objective_metric.sign >= 0),
                                  color=table_values_columns[-1][1:]),
                        dimensions=pcc_dims)],
                    layout={})
                task_logger.report_plotly(
                    title='Parallel Coordinates', series='',
                    iteration=0, figure=plotly_pcc)
            if force:
                task = self._task or Task.current_task()
                if task:
                    task.upload_artifact(name='summary', artifact_object={'table': table_values})
    clearml.automation.HyperParameterOptimizer._report_completed_status = patchFunc

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
