import io
from unittest import TestCase, mock

from tools.trainer import Logger


@mock.patch('time.monotonic', mock.MagicMock(return_value=42))
class TestLogger(TestCase):

    def log_some_data(self, header, acc_odd):
        logger = Logger()
        with io.StringIO() as out:
            for i in logger.log_every(range(20), 2, "train", header=header,
                                      metrics_format="loss:{loss:.3f}, accuracy:{accuracy}", file=out):
                logger.update(loss=i / 100)
                if i % 2 == acc_odd:
                    logger.update(accuracy=i / 20)
            return out.getvalue()

    def test_logger_no_header(self):
        s = self.log_some_data("", acc_odd=0)
        self.assertEqual(s, (
            "train	[ 2/20]	(0:00:00<=0:00:00s)	loss:0.010, accuracy:None\n"
            "train	[ 4/20]	(0:00:00<=0:00:00s)	loss:0.030, accuracy:None\n"
            "train	[ 6/20]	(0:00:00<=0:00:00s)	loss:0.050, accuracy:None\n"
            "train	[ 8/20]	(0:00:00<=0:00:00s)	loss:0.070, accuracy:None\n"
            "train	[10/20]	(0:00:00<=0:00:00s)	loss:0.090, accuracy:None\n"
            "train	[12/20]	(0:00:00<=0:00:00s)	loss:0.110, accuracy:None\n"
            "train	[14/20]	(0:00:00<=0:00:00s)	loss:0.130, accuracy:None\n"
            "train	[16/20]	(0:00:00<=0:00:00s)	loss:0.150, accuracy:None\n"
            "train	[18/20]	(0:00:00<=0:00:00s)	loss:0.170, accuracy:None\n"
            "train	[20/20]	(0:00:00<=0:00:00s)	loss:0.190, accuracy:None\n"
            " Total time: 0:00:00 (0.0000 s / it)\n"))

    def test_logger_with_header(self):
        s = self.log_some_data("Evaluating model", acc_odd=1)
        self.assertEqual(s, (
            "------------------------------\n"
            "Evaluating model\n"
            "------------------------------\n"
            "train	[ 2/20]	(0:00:00<=0:00:00s)	loss:0.010, accuracy:0.05\n"
            "train	[ 4/20]	(0:00:00<=0:00:00s)	loss:0.030, accuracy:0.15\n"
            "train	[ 6/20]	(0:00:00<=0:00:00s)	loss:0.050, accuracy:0.25\n"
            "train	[ 8/20]	(0:00:00<=0:00:00s)	loss:0.070, accuracy:0.35\n"
            "train	[10/20]	(0:00:00<=0:00:00s)	loss:0.090, accuracy:0.45\n"
            "train	[12/20]	(0:00:00<=0:00:00s)	loss:0.110, accuracy:0.55\n"
            "train	[14/20]	(0:00:00<=0:00:00s)	loss:0.130, accuracy:0.65\n"
            "train	[16/20]	(0:00:00<=0:00:00s)	loss:0.150, accuracy:0.75\n"
            "train	[18/20]	(0:00:00<=0:00:00s)	loss:0.170, accuracy:0.85\n"
            "train	[20/20]	(0:00:00<=0:00:00s)	loss:0.190, accuracy:0.95\n"
            "Evaluating model Total time: 0:00:00 (0.0000 s / it)\n"))

    def test_custom_section(self):
        logger = Logger()
        with io.StringIO() as out:
            custom_section = logger.start_section("custom", "Custom log block")
            custom_section.print_header(file=out)
            for i in range(10):
                custom_section.step_end()
                custom_section.print_row(file=out)
            custom_section.print_footer(file=out)
            s = out.getvalue()
        self.assertEqual(s, (
            "------------------------------\n"
            "Custom log block\n"
            "------------------------------\n"
            "custom	[1/?]	(0:00:00s)\n"
            "custom	[2/?]	(0:00:00s)\n"
            "custom	[3/?]	(0:00:00s)\n"
            "custom	[4/?]	(0:00:00s)\n"
            "custom	[5/?]	(0:00:00s)\n"
            "custom	[6/?]	(0:00:00s)\n"
            "custom	[7/?]	(0:00:00s)\n"
            "custom	[8/?]	(0:00:00s)\n"
            "custom	[9/?]	(0:00:00s)\n"
            "custom	[10/?]	(0:00:00s)\n"
            "Custom log block Total time: 0:00:00 (0.0000 s / it)\n"))

    def test_logger_persistence(self):
        logger = Logger()
        with io.StringIO() as silent:
            for i in logger.log_every(range(20), 2, "train",
                                      metrics_format="loss:{loss:.3f}, accuracy:{accuracy}", file=silent):
                logger.update(loss=i / 100)
                if i % 2 == 1:
                    logger.update(accuracy=i / 20)

            for i in logger.log_every(range(10), 2, "eval", header="Evaluating model",
                                      metrics_format="loss:{loss:.3f}, accuracy:{accuracy}", file=silent):
                logger.update(loss=i / 100)
                if i % 2 == 0:
                    logger.update(accuracy=i / 20)

            custom_section = logger.start_section("custom", "Custom log block")
            for i in range(10):
                custom_section.step_end()

        saved_state = logger.state_dict()

        loaded_logger = Logger.from_dict(saved_state)
        with io.StringIO() as out:
            loaded_logger.print_logs(2, file=out)
            s = out.getvalue()
            self.assertEqual(s, (
                "train	[ 2/20]	(0:00:00<=0:00:00s)	loss:0.010, accuracy:0.05\n"
                "train	[ 4/20]	(0:00:00<=0:00:00s)	loss:0.030, accuracy:0.15\n"
                "train	[ 6/20]	(0:00:00<=0:00:00s)	loss:0.050, accuracy:0.25\n"
                "train	[ 8/20]	(0:00:00<=0:00:00s)	loss:0.070, accuracy:0.35\n"
                "train	[10/20]	(0:00:00<=0:00:00s)	loss:0.090, accuracy:0.45\n"
                "train	[12/20]	(0:00:00<=0:00:00s)	loss:0.110, accuracy:0.55\n"
                "train	[14/20]	(0:00:00<=0:00:00s)	loss:0.130, accuracy:0.65\n"
                "train	[16/20]	(0:00:00<=0:00:00s)	loss:0.150, accuracy:0.75\n"
                "train	[18/20]	(0:00:00<=0:00:00s)	loss:0.170, accuracy:0.85\n"
                "train	[20/20]	(0:00:00<=0:00:00s)	loss:0.190, accuracy:0.95\n"
                " Total time: 0:00:00 (0.0000 s / it)\n"
                "------------------------------\n"
                "Evaluating model\n"
                "------------------------------\n"
                "eval	[ 2/10]	(0:00:00<=0:00:00s)	loss:0.010, accuracy:None\n"
                "eval	[ 4/10]	(0:00:00<=0:00:00s)	loss:0.030, accuracy:None\n"
                "eval	[ 6/10]	(0:00:00<=0:00:00s)	loss:0.050, accuracy:None\n"
                "eval	[ 8/10]	(0:00:00<=0:00:00s)	loss:0.070, accuracy:None\n"
                "eval	[10/10]	(0:00:00<=0:00:00s)	loss:0.090, accuracy:None\n"
                "Evaluating model Total time: 0:00:00 (0.0000 s / it)\n"
                "------------------------------\n"
                "Custom log block\n"
                "------------------------------\n"
                "custom	[2/?]	(0:00:00s)\n"
                "custom	[4/?]	(0:00:00s)\n"
                "custom	[6/?]	(0:00:00s)\n"
                "custom	[8/?]	(0:00:00s)\n"
                "custom	[10/?]	(0:00:00s)\n"
                "Custom log block Total time: 0:00:00 (0.0000 s / it)\n"))
