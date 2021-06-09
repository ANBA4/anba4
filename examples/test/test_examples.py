import os
import unittest

thisdir = os.path.dirname(os.path.realpath(__file__))
examples_dir = os.path.dirname(thisdir)
all_examples = os.listdir(examples_dir)

class TestExamples(unittest.TestCase):

    def testAllExamplesRun(self):
        for f in all_examples:
            if (f not in ['QuadMesh.py']) and ('.py' in f):
                # Go to location due to relative path use for airfoil files
                print('\n\n')
                print('NOW RUNNING:',f)
                print()
                try:
                    exec(open(os.path.join(examples_dir, f)).read())
                except:
                    self.assertTrue(False)
            else:
                print('\n\n')
                print('SKIPPING FILE:',f)
                print()

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestExamples))
    return suite

if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)