from unittest import TestCase
import numpy as np
import utils

class TestUtils(TestCase):

    def test_fileload(self):
        os_data_file = "cleanData/ah_filteredData_OS.mat"
        os_df = utils.load_individual_os_data(os_data_file)
        self.assertTrue(np.all(os_df[os_df.Orientation==-1]['MaskContrast']==0))
        self.assertTrue(np.all(os_df[os_df.Orientation==-2]['MaskContrast']==0))

    def test_grouping(self):
        os_data_file = "cleanData/ah_filteredData_OS.mat"
        os_df = utils.load_individual_os_data(os_data_file)
        gvars = ["Subject", "Eye", "Orientation", "Presentation"] # Presentation conditions
        gvars_mask = gvars + ["MaskContrast"] # Mask contrast, m in the model
        gvars_masktarget = gvars_mask + ["ProbeContrastUsed"] # Target contrast, t in the model
        grouped, condensed_df = utils.summarize_conditions(os_df, gvars_masktarget)
        for gv, g in grouped:
            self.assertTrue(np.all(g.Eye==g.Eye.iloc[0])) # Make sure we only are looking at data for one eye
            self.assertTrue(np.all(g.Presentation==g.Presentation.iloc[0])) # again, one condition only
            
