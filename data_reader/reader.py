class CsvReader(object):
    def __init__(self, file_path):
        self.__file_path = file_path

    #titanic train data set
    def get_titanic_train_data(self):
        titanic_features, titanic_labels = [], []

        with open(self.__file_path, 'r') as reader:
            all_lines = reader.readlines()

            for line in all_lines[1:]:
                line_tokens = line.strip().split(',')

                # iris_features.append(list(map(lambda v: float(v), line_tokens[1:-1])))
                if line_tokens[5] == 'female':
                    line_tokens[5] = '0'
                elif line_tokens[5] == 'male':
                    line_tokens[5] = '1'
                if line_tokens[6] == '':
                    line_tokens[6] = '20'
                if line_tokens[10] == '':
                    line_tokens[10] = '8'
                if line_tokens[6].find('.') != -1:
                    line_tokens[6] = line_tokens[6][0:line_tokens[6].find('.')]
                if line_tokens[10].find('.') != -1:
                    line_tokens[10] = line_tokens[10][0:line_tokens[10].find('.')]
                line_tokens[6] = float(line_tokens[6]) * 0.01;
                titanic_features.append(list(map(lambda v: float(v), [line_tokens[2], line_tokens[5], line_tokens[6]])))
                # iris_labels.append(str(line_tokens[-1]))
                #titanic_labels.append(line_tokens[1])
                #titanic_labels.append(map(lambda v: float(v), [line_tokens[1]]))
                titanic_labels.append(float(line_tokens[1]))

        return titanic_features, titanic_labels

    #titanic test data set
    def get_titanic_test_data(self):
        titanic_test_features = []

        with open(self.__file_path, 'r') as reader:
            all_lines = reader.readlines()

            for line in all_lines[1:]:
                line_tokens = line.strip().split(',')

                # iris_features.append(list(map(lambda v: float(v), line_tokens[1:-1])))
                if line_tokens[4] == 'female':
                    line_tokens[4] = '0'
                elif line_tokens[4] == 'male':
                    line_tokens[4] = '1'
                if line_tokens[5] == '':
                    line_tokens[5] = '20'
                if line_tokens[9] == '':
                    line_tokens[9] = '8'
                if line_tokens[5].find('.') != -1:
                    line_tokens[5] = line_tokens[5][0:line_tokens[5].find('.')]
                if line_tokens[9].find('.') != -1:
                    line_tokens[9] = line_tokens[9][0:line_tokens[9].find('.')]
                line_tokens[5] = float(line_tokens[5]) * 0.01
                titanic_test_features.append(list(map(lambda v: float(v), [line_tokens[1], line_tokens[4], line_tokens[5]])))
                # iris_labels.append(str(line_tokens[-1]))
                #titanic_test_labels.append(line_tokens[1])
                #titanic_test_labels.append(list(map(lambda v: float(v), [line_tokens[1]])))

        return titanic_test_features


