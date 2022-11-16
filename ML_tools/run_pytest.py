# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:28:32 2022

@author: aossipov
"""

import pytest
import sys

test_script_name = 'test_train_validation_test.py'

sys.exit(pytest.main(['--disable-pytest-warnings', test_script_name])) 
