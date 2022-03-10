class Flower(object):
    def __init__(self, floral=None, leaf=None):
        self.floral = floral
        self.__leaf = leaf
        
    def flowing(self):
        print("flower")
        
    def __grow(self):
        print 'grow grow'