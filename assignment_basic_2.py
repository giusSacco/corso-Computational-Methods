'''Module: basic Python
Assignment #2 (Set 30, 2021)

--- Goal
Write a program to explore the properties of a few elementary Particles.
The program must contain a Base class Particle and two Child classes, Proton and Alpha, that inherit from it.

--- Specifications
- instances of the class Particle must be initialized with their mass, charge, and name
- the class constructor must also accept (optionally) and store one and only one of the following quantities: energy, momentum, beta or gamma
- whatever the choice, the user should be able to read and set any of the
  above mentioned quantites using just the '.' (dot) operator e.g.
  print(my_particle.energy), my_particle.beta = 0.5
- attempts to set non physical values should be rejected
- the Particle class must have a method to print the Particle information in
  a formatted way
- the child classes Alpha and Protons must use class attributes to store their mass, charge and name'''

import math

class Particle():
    def __init__(self, mass, charge, name, momentum = 0.):
        self._mass = mass
        self._charge = charge
        self._name = name
        self.momentum = momentum

    @property
    def mass(self):
        return self._mass

    @property
    def charge(self):
        return self._charge

    @property
    def name(self):
        return self._name

    @property
    def energy(self):
        return self.momentum**2/(2*self.mass)

    @energy.setter
    def energy(self, value):
        if value < 0:
            print('Che stai a fa?')
            return
        self.momentum = math.sqrt(2*self.mass*value)

    def __str__(self) -> str:
        return f'Particle is {self.name}, mass {self.mass}, charge {self.charge}, momentum {self.momentum}, energy {self.energy}'

class Proton(Particle):

    MASS = 0.511
    CHARGE = -1.
    NAME = 'p'

    def __init__(self, momentum = 0.):
        super().__init__(mass = self.MASS, charge = self.CHARGE, name = self.NAME, momentum = momentum)

p = Proton(momentum = 25)
p.energy = 0.1
print(p)