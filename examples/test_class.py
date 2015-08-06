__author__ = 'NLP-PC'


class Dog(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

    def change_name(self, name):
        self.name = name

    def change_gender(self, gender):
        self.gender = gender


d = Dog('xiao', 23)
print(d.name, d.gender)
d.change_gender('xiaoo')
d.change_name('feaaa')
print(d.name, d.gender)
