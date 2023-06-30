import torch

class CoresetSelection(object):
    @staticmethod
    def random_selection(total_num, num):
        print('Random selection.')
        score_random_index = torch.randperm(total_num)

        return score_random_index[:int(num)]

    @staticmethod
    def score_monotonic_selection(data_score, key, ratio, descending, class_balanced=False):
        score = data_score[key]
        score_sorted_index = score.argsort(descending=descending)
        total_num = ratio * data_score['targets'].shape[0]

        if class_balanced:
            print('Class balance mode.')
            all_index = torch.arange(data_score['targets'].shape[0])
            #Permutation
            targets_list = data_score['targets'][score_sorted_index]
            targets_unique = torch.unique(targets_list)
            # print(targets_unique)

            target_nums = []
            for target in targets_unique:
                target_index_mask = (targets_list == target)
                targets_num = target_index_mask.sum()
                target_nums.append(targets_num)
                # print("target, target_num", target, targets_num)
            

            # print("targets num", targets_num)
            #Guarantee the class ratio doesn't change
            selected_index = []
            for i, target in enumerate(targets_unique):
                target_index_mask = torch.flatten((targets_list == target))
                target_index = all_index[target_index_mask]
                target_coreset_num = target_nums[i] * ratio
                selected_index = selected_index + list(target_index[:int(target_coreset_num)])
            selected_index = torch.tensor(selected_index)
            print(f'High priority {key}: {score[score_sorted_index[selected_index][:15]]}')
            print(f'Low priority {key}: {score[score_sorted_index[selected_index][-15:]]}')

            return score_sorted_index[selected_index]

        else:
            print(f'High priority {key}: {score[score_sorted_index[:15]]}')
            print(f'Low priority {key}: {score[score_sorted_index[-15:]]}')
            return score_sorted_index[:int(total_num)]