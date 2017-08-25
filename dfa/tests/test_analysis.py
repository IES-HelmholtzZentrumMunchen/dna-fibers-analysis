"""
Tests of the analysis module of the DNA fiber analysis package.
"""

import unittest
from dfa import analysis as ana
from os import path as op


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        data_path = 'dfa/tests/data'

        self.profiles = {
            1: ana.np.loadtxt(op.join(data_path, 'profiles_1.csv'),
                              delimiter=',', skiprows=1, usecols=(0, 1, 2)),
            2: ana.np.loadtxt(op.join(data_path, 'profiles_2.csv'),
                              delimiter=',', skiprows=1, usecols=(0, 1, 2)),
            3: ana.np.loadtxt(op.join(data_path, 'profiles_3.csv'),
                              delimiter=',', skiprows=1, usecols=(0, 1, 2)),
            4: ana.np.loadtxt(op.join(data_path, 'profiles_4.csv'),
                              delimiter=',', skiprows=1, usecols=(0, 1, 2)),
            5: ana.np.loadtxt(op.join(data_path, 'profiles_5.csv'),
                              delimiter=',', skiprows=1, usecols=(0, 1, 2)),
            6: ana.np.loadtxt(op.join(data_path, 'profiles_6.csv'),
                              delimiter=',', skiprows=1, usecols=(0, 1, 2))}

        self.possible_patterns = {
            1: [(803.54064601447806, [], [1]),
                (421.93623251601588, [340, 640], [1, 0, 1]),
                (682.06700340680618, [340], [1, 0])],
            2: [(775.33204476332719, [], [1]),
                (126.33900709126601, [122, 488], [0, 1, 0]),
                (509.89617466361108, [488], [1, 0])],
            3: [(73.710506176943525, [], [0]),
                (62.686511537653885, [3, 12], [1, 0, 1]),
                (62.658494733780167, [12], [0, 1])],
            4: [(37.833424801896577, [], [1]),
                (38.880827208750034, [4, 14], [1, 0, 1]),
                (37.902299310431289, [14], [0, 1])],
            5: [(196.67130963774542, [], [1]),
                (32.877788570204274, [41, 128], [0, 1, 0]),
                (47.715693666606064, [41], [0, 1])],
            6: [(177.04777603197672, [], [1]),
                (37.281005068208501, [100], [1, 0])]}

        self.selected_patterns = {
            (1, 0, 0): (421.93623251601588, [340, 640], [1, 0, 1]),
            (2, 0, 0): (126.33900709126601, [122, 488], [0, 1, 0]),
            (3, 0, 0): (62.658494733780167, [12], [0, 1]),
            (4, 0, 0): (37.83342480189658, [], [1]),
            (5, 0, 0): (32.877788570204274, [41, 128], [0, 1, 0]),
            (6, 0, 0): (37.281005068208501, [100], [1, 0]),
            (1, 0, 20): (432.76737198904402, [340, 640], [1, 0, 1]),
            (2, 0, 20): (130.96727964178041, [122, 488], [0, 1, 0]),
            (3, 0, 20): (73.71050617694353, [], [0]),
            (4, 0, 20): (37.83342480189658, [], [1]),
            (5, 0, 20): (38.579879947373058, [41, 128], [0, 1, 0]),
            (6, 0, 20): (44.14438173979034, [100], [1, 0]),
            (1, 0, 50): (449.01408119858627, [340, 640], [1, 0, 1]),
            (2, 0, 50): (137.909688467552, [122, 488], [0, 1, 0]),
            (3, 0, 50): (73.71050617694353, [], [0]),
            (4, 0, 50): (37.83342480189658, [], [1]),
            (5, 0, 50): (47.133017013126235, [41, 128], [0, 1, 0]),
            (6, 0, 50): (54.439446747163096, [100], [1, 0]),
            (1, 0, 80): (465.26079040812846, [340, 640], [1, 0, 1]),
            (2, 0, 80): (144.8520972933236, [122, 488], [0, 1, 0]),
            (3, 0, 80): (73.71050617694353, [], [0]),
            (4, 0, 80): (37.83342480189658, [], [1]),
            (5, 0, 80): (55.686154078879404, [41, 128], [0, 1, 0]),
            (6, 0, 80): (64.734511754535845, [100], [1, 0]),
            (1, 20, 0): (421.98862725121319, [340, 640], [1, 0, 1]),
            (2, 20, 0): (126.37755781710898, [122, 488], [0, 1, 0]),
            (3, 20, 0): (62.658494733780167, [12], [0, 1]),
            (4, 20, 0): (37.83342480189658, [], [1]),
            (5, 20, 0): (37.681824032664736, [41, 128], [0, 1, 0]),
            (6, 20, 0): (37.281005068208501, [100], [1, 0]),
            (1, 20, 20): (432.81976672424133, [340, 640], [1, 0, 1]),
            (2, 20, 20): (131.00583036762336, [122, 488], [0, 1, 0]),
            (3, 20, 20): (73.71050617694353, [], [0]),
            (4, 20, 20): (37.83342480189658, [], [1]),
            (5, 20, 20): (43.38391540983352, [41, 128], [0, 1, 0]),
            (6, 20, 20): (44.14438173979034, [100], [1, 0]),
            (1, 20, 50): (449.06647593378358, [340, 640], [1, 0, 1]),
            (2, 20, 50): (137.94823919339495, [122, 488], [0, 1, 0]),
            (3, 20, 50): (73.71050617694353, [], [0]),
            (4, 20, 50): (37.83342480189658, [], [1]),
            (5, 20, 50): (51.93705247558669, [41, 128], [0, 1, 0]),
            (6, 20, 50): (54.439446747163096, [100], [1, 0]),
            (1, 20, 80): (465.31318514332582, [340, 640], [1, 0, 1]),
            (2, 20, 80): (144.89064801916655, [122, 488], [0, 1, 0]),
            (3, 20, 80): (73.71050617694353, [], [0]),
            (4, 20, 80): (37.83342480189658, [], [1]),
            (5, 20, 80): (60.490189541339866, [41, 128], [0, 1, 0]),
            (6, 20, 80): (64.734511754535845, [100], [1, 0]),
            (1, 50, 0): (422.06721935400918, [340, 640], [1, 0, 1]),
            (2, 50, 0): (126.43538390587341, [122, 488], [0, 1, 0]),
            (3, 50, 0): (62.658494733780167, [12], [0, 1]),
            (4, 50, 0): (37.83342480189658, [], [1]),
            (5, 50, 0): (44.887877226355421, [41, 128], [0, 1, 0]),
            (6, 50, 0): (37.281005068208501, [100], [1, 0]),
            (1, 50, 20): (432.89835882703733, [340, 640], [1, 0, 1]),
            (2, 50, 20): (131.0636564563878, [122, 488], [0, 1, 0]),
            (3, 50, 20): (73.71050617694353, [], [0]),
            (4, 50, 20): (37.83342480189658, [], [1]),
            (5, 50, 20): (50.589968603524206, [41, 128], [0, 1, 0]),
            (6, 50, 20): (44.14438173979034, [100], [1, 0]),
            (1, 50, 50): (449.14506803657957, [340, 640], [1, 0, 1]),
            (2, 50, 50): (138.0060652821594, [122, 488], [0, 1, 0]),
            (3, 50, 50): (73.71050617694353, [], [0]),
            (4, 50, 50): (37.83342480189658, [], [1]),
            (5, 50, 50): (57.387209574342457, [41], [0, 1]),
            (6, 50, 50): (54.439446747163096, [100], [1, 0]),
            (1, 50, 80): (465.39177724612182, [340, 640], [1, 0, 1]),
            (2, 50, 80): (144.94847410793099, [122, 488], [0, 1, 0]),
            (3, 50, 80): (73.71050617694353, [], [0]),
            (4, 50, 80): (37.83342480189658, [], [1]),
            (5, 50, 80): (63.190119118984292, [41], [0, 1]),
            (6, 50, 80): (64.734511754535845, [100], [1, 0]),
            (1, 80, 0): (422.14581145680518, [340, 640], [1, 0, 1]),
            (2, 80, 0): (126.49320999463785, [122, 488], [0, 1, 0]),
            (3, 80, 0): (62.658494733780167, [12], [0, 1]),
            (4, 80, 0): (37.83342480189658, [], [1]),
            (5, 80, 0): (47.715693666606064, [41], [0, 1]),
            (6, 80, 0): (37.281005068208501, [100], [1, 0]),
            (1, 80, 20): (432.97695092983332, [340, 640], [1, 0, 1]),
            (2, 80, 20): (131.12148254515225, [122, 488], [0, 1, 0]),
            (3, 80, 20): (73.71050617694353, [], [0]),
            (4, 80, 20): (37.83342480189658, [], [1]),
            (5, 80, 20): (51.584300029700621, [41], [0, 1]),
            (6, 80, 20): (44.14438173979034, [100], [1, 0]),
            (1, 80, 50): (449.22366013937557, [340, 640], [1, 0, 1]),
            (2, 80, 50): (138.06389137092384, [122, 488], [0, 1, 0]),
            (3, 80, 50): (73.71050617694353, [], [0]),
            (4, 80, 50): (37.83342480189658, [], [1]),
            (5, 80, 50): (57.387209574342457, [41], [0, 1]),
            (6, 80, 50): (54.439446747163096, [100], [1, 0]),
            (1, 80, 80): (465.47036934891781, [340, 640], [1, 0, 1]),
            (2, 80, 80): (145.00630019669543, [122, 488], [0, 1, 0]),
            (3, 80, 80): (73.71050617694353, [], [0]),
            (4, 80, 80): (37.83342480189658, [], [1]),
            (5, 80, 80): (63.190119118984292, [41], [0, 1]),
            (6, 80, 80): (64.734511754535845, [100], [1, 0])}

        self.models = {
            1: ('2nd label termination', [48.24829, 42.57202, 43.56537]),
            2: ('1st label origin', [17.31262, 51.93787, 18.16406]),
            3: ('2nd label origin', [25.68512]),
            4: ('stalled fork/1st label termination', [12.49389362]),
            5: ('1st label origin', [5.8210187, 12.35191679, 0.85185623]),
            6: ('ongoing fork', [100., 88.])}

        self.data = ana.pd.DataFrame({
            'profile': [0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5],
            'pattern': ['2nd label termination', '2nd label termination',
                        '2nd label termination', '1st label origin',
                        '1st label origin', '1st label origin',
                        '2nd label origin',
                        'stalled fork/1st label termination',
                        '1st label origin', '1st label origin',
                        '1st label origin', 'ongoing fork', 'ongoing fork'],
            'channel': ['IdU', 'CIdU', 'IdU', 'CIdU', 'IdU', 'CIdU', 'CIdU',
                        'IdU', 'CIdU', 'IdU', 'CIdU', 'IdU', 'CIdU'],
            'length': [48.248290, 42.572020, 43.565370, 17.312620, 51.937870,
                       18.164060, 25.685120, 12.493894, 5.821019, 12.351917,
                       0.851856, 100., 88.]},
            columns=['pattern', 'channel', 'length'],
            index=ana.pd.Int64Index(
                [0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5], name='profile'))

        self.fork_rate = ana.pd.Series(
            data=[1.049180308930711, 6.8333346578917489], index=[1, 4],
            name='Fork rate')

        self.fork_speed = ana.pd.Series(
            data=[88.], index=[5], name='Fork speed')

        self.patterns = ana.pd.Series(
            data=['2nd label termination', '1st label origin',
                  '2nd label origin', 'stalled fork/1st label termination',
                  '1st label origin', 'ongoing fork'],
            index=[0, 1, 2, 3, 4, 5], name='Patterns')

    def tearDown(self):
        pass

    @staticmethod
    def get_x_and_y(profiles):
        return profiles.T[0], \
               ana.np.log(profiles.T[2]) - ana.np.log(profiles.T[1])

    def test__select_possible_patterns(self):
        for index in self.profiles.keys():
            x, y = TestAnalysis.get_x_and_y(self.profiles[index])
            possible_patterns = ana._select_possible_patterns(x, y)

            for actual_pattern, expected_pattern in \
                    zip(possible_patterns, self.possible_patterns[index]):
                self.assertAlmostEqual(
                    actual_pattern[0], expected_pattern[0],
                    msg='profiles #{}'.format(index))
                self.assertListEqual(
                    actual_pattern[1], expected_pattern[1],
                    msg='profiles #{}'.format(index))
                self.assertListEqual(
                    actual_pattern[2], expected_pattern[2],
                    msg='profiles #{}'.format(index))

    def test__choose_pattern(self):
        keys = list(zip(*self.selected_patterns.keys()))

        for discrepancy in ana.np.unique(keys[1]):
            for contrast in ana.np.unique(keys[2]):
                for index in ana.np.unique(keys[0]):
                    x, y = TestAnalysis.get_x_and_y(self.profiles[index])
                    selected_pattern = ana._choose_pattern(
                        self.possible_patterns[index], x, y,
                        discrepancy, contrast)

                    self.assertAlmostEqual(
                        selected_pattern[0],
                        self.selected_patterns[index, discrepancy, contrast][0],
                        msg='profiles #{}, d={}, c={}'.format(index,
                                                              discrepancy,
                                                              contrast))
                    self.assertListEqual(
                        selected_pattern[1],
                        self.selected_patterns[index, discrepancy, contrast][1],
                        msg='profiles #{}, d={}, c={}'.format(index,
                                                              discrepancy,
                                                              contrast))
                    self.assertListEqual(
                        selected_pattern[2],
                        self.selected_patterns[index, discrepancy, contrast][2],
                        msg='profiles #{}, d={}, c={}'.format(index,
                                                              discrepancy,
                                                              contrast))

    def test_analyze(self):
        for index in self.profiles.keys():
            pattern, length = ana.analyze(
                self.profiles[index], discrepancy=30, contrast=80)

            self.assertEqual(pattern['name'], self.models[index][0])
            ana.np.testing.assert_allclose(length, self.models[index][1])

    def get_data(self):
        return ana.analyzes(
            list(self.profiles.values()),
            update_model=False, discrepancy=30, contrast=80)

    def test_analyzes(self):
        data = self.get_data()
        self.assertTrue(
            data.index.equals(self.data.index) and
            data['pattern'].equals(self.data['pattern']) and
            data['channel'].equals(self.data['channel']) and
            ana.np.isclose(data['length'], self.data['length']).all(),
            msg='\n\nactual data is:\n {}\n\n'
                'expected data is:\n {}\n\n'.format(data,
                                                    self.data))

    def test_fork_speed(self):
        data = self.get_data()
        ana.fork_speed(data).equals(self.fork_speed)

    def test_fork_rate(self):
        data = self.get_data()
        ana.fork_rate(data).equals(self.fork_rate)

    def test_get_patterns(self):
        data = self.get_data()
        ana.get_patterns(data).equals(self.patterns)
