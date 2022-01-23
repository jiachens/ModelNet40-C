import csv
import os

class RecordExp:
    def __init__(self, file_name):
        self.file_name = file_name
        self.param_recorded = False
        self.result_recorded = False
        self.param_dict = {}

    def record_param(self, param_dict):
        """
        all parameters must be given at the same time. parameters must be given before the results
        :return:
        """
        assert not self.param_recorded
        self.param_recorded = True
        self.param_dict = param_dict

    def record_result(self, result_dict):
        """
        all results must be given at the same time
        :return:
        """
        assert self.param_recorded
        assert not self.result_recorded
        self.result_recorded = True
        assert len(set(result_dict.keys()) & set(self.param_dict.keys())) == 0

        if os.path.exists(self.file_name):
            with open(self.file_name, 'r') as csv_file:
                reader = csv.reader(csv_file)
                fields = next(reader)
        else:
            print("This is the first record of the experiment")
            fields = list(self.param_dict.keys()) + list(result_dict.keys())
            with open(self.file_name, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(fields)

        self.param_dict.update(result_dict)

        values = []
        for field in fields:
            if field in self.param_dict:
                values.append(self.param_dict[field])
            else:
                values.append("")

        with open(self.file_name, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(values)


def get_mv_mean_var(param_tuple):
    """
    :param param_tuple: should be of the following form
        ('dataset', "modelnet), ('views', 1), ('resolution', 128), ('trans', -1.4), ('size', 4), ('normalize', False)
    :return:
    """
    data = {
        (
            ('dataset', 'object'),
            ('views', 6),
            ('resolution', 128),
            ('trans', -1.4),
            ('size', 1),
            ('normalize', False),
            ('norm_pc', True)
        ): [
            (0.04440825, 0.0615424),
            (0.04496584, 0.06237658),
            (0.044289585, 0.061655156),
            (0.044222, 0.061538428)
        ],
        (
            ('dataset', 'modelnet'),
            ('views', 6),
            ('resolution', 128),
            ('trans', -1.4),
            ('size', 1),
            ('normalize', False),
            ('norm_pc', True)
        ): [
            (0.06295275, 0.086910926),
            (0.06327734, 0.087433286),
            (0.06296529, 0.08695659),
            (0.062923886, 0.086918436)
        ],
    }

    mean_var_list = data[param_tuple]
    mean_list = [x for x, y in mean_var_list]
    var_list = [y for x, y in mean_var_list]
    mean = sum(mean_list) / len(mean_list)
    var = sum(var_list) / len(var_list)
    return mean, var
