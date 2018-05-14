from unittest import TestCase
import numpy as np
import two_stage_model as model

class TestUtils(TestCase):

    def test_loglikelihood(self):
        self.assertEquals(model.loglikelihood(10, 10, 1), 0)
        self.assertEquals(model.loglikelihood(10, 0, 0), 0)
        self.assertEquals(model.loglikelihood(10, 0, 1), 0)
        self.assertEquals(model.loglikelihood(10, 10, 0), 0)
        self.assertEquals(model.loglikelihood(10, 6, .6), 0)
        self.assertEquals(model.loglikelihood(10, 4, .4), 0)
            
