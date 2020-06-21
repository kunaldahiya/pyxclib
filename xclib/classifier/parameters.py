from .parameters_base import ParametersBase


class Parameters(ParametersBase):
    """
        Parameter class for XML Classifiers
    """
    def __init__(self, description):
        super().__init__(description)
        self._construct()

    def _construct(self):
        super()._construct()
        self.parser.add_argument(
            '--num_threads',
            dest='num_threads',
            default=4,
            action='store',
            type=int,
            help='HSNW params')
        self.parser.add_argument(
            '--max_iter',
            dest='max_iter',
            default=100,
            action='store',
            type=int,
            help='#iterations in SVM'),
        self.parser.add_argument(
            '--dual',
            dest='dual',
            action='store',
            default=False,
            type=bool,
            help='Solve problem in primal/dual'),
        self.parser.add_argument(
            '-mode',
            dest='mode',
            action='store',
            type=str,
            help='train/predict'),
        self.parser.add_argument(
            '-clf_type',
            dest='clf_type',
            action='store',
            type=str,
            help='Classifier type'),
        self.parser.add_argument(
            '--M',
            dest='M',
            default=100,
            action='store',
            type=int,
            help='M')
        self.parser.add_argument(
            '--efC',
            dest='efC',
            default=300,
            action='store',
            type=int,
            help='efC')
        self.parser.add_argument(
            '--efS',
            dest='efS',
            default=300,
            action='store',
            type=int,
            help='efS')
        self.parser.add_argument(
            '--num_neighbours',
            dest='num_neighbours',
            default=300,
            action='store',
            type=int,
            help='num_neighbours')
        self.parser.add_argument(
            '--batch_size',
            dest='batch_size',
            default=100,
            action='store',
            type=int,
            help='batch size')
        self.parser.add_argument(
            '--tol',
            dest='tol',
            default=0.01,
            type=float,
            action='store',
            help='Tolerance in SVM training')
        self.parser.add_argument(
            '--beta',
            dest='beta',
            default=0.2,
            type=float,
            action='store',
            help='weightage of clf score')
        self.parser.add_argument(
            '--start_index',
            dest='start_index',
            default=0,
            type=int,
            action='store',
            help='Start training from here.')
        self.parser.add_argument(
            '--end_index',
            dest='end_index',
            default=-1,
            type=int,
            action='store',
            help='Finish training from here.')
        self.parser.add_argument(
            '--C',
            dest='C',
            default=1.0,
            type=float,
            action='store',
            help='C')
        self.parser.add_argument(
            '--threshold',
            dest='threshold',
            default=0.01,
            type=float,
            action='store',
            help='threshold for post L1')
        self.parser.add_argument(
            '--norm',
            dest='norm',
            default=None,
            type=str,
            action='store',
            help='Normalize data')
        self.parser.add_argument(
            '--use_shortlist',
            action='store_true',
            help='Use shortlist')
