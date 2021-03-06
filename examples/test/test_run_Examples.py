import unittest
import numpy as np    
import glob
import os

def execfile(filepath, globals=None, locals=None):
    """ Execute a given python file """
    if globals is None:
        globals = {"__name__": "__main__"}
    globals.update({
        "__file__": filepath,
    })
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), globals, locals)

class TestExamples(unittest.TestCase):
    def test_run_examples(self):
        exclude_list=[]
        # Add tests to class
        MyDir=os.path.dirname(__file__)
        files = glob.glob(os.path.join(MyDir,'../*.py'))
        for f in files:
            print('\n--------------------------------------------------------------')
            if ('QuadMesh.py' not in f) and ('.py' in f):
                print('Running example script: {}'.format(f))
                if hasattr(self,'subTest'):
                    with self.subTest(filename=os.path.basename(f)):
                        execfile(f, {'__name__': '__test__', 'print': lambda *_:None})
            else:
                print('Skip file:',f)


if __name__ == '__main__':
    unittest.main()
