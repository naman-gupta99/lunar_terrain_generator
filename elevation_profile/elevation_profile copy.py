import numpy as np

class CraterHeightProfile:
    def __init__(self, D):
        self.D = D
        self.H0 = 0.196 * D**1.01
        self.Hr0 = 0.036 * D**1.01

    def calculate_parameters(self, H, Hr, Tr, Pr, Emin):
        self.H = H
        self.Hr = Hr
        self.Tr = Tr
        self.Pr = Pr
        self.Emin = Emin

        self.alpha = (Hr + Tr - Pr - self.Hr0) / (self.Hr0 - Hr + H)
        self.beta = 3 * (Hr + Tr - Pr - self.Hr0) / (2 * (self.Hr0 - Hr + H))

    def get_height(self, r):
        x = (2 * r / self.D) - 1
        
        if -1 <= x < self.alpha:
            return (self.Hr0 - self.Hr + self.H) * x**2 + 2 * (self.Hr0 - self.Hr + self.H) * x + self.Hr0
        elif self.alpha <= x <= 0:
            return ((self.Hr0 - self.Hr + self.H) * (x + 1) / self.alpha) * x**2 + self.Hr + self.Tr - self.Pr
        elif 0 < x <= self.beta:
            return (-2 * (self.Hr0 - self.Hr + self.H) / (3 * self.beta**2)) * x**3 + \
                   ((self.Hr0 - self.Hr + self.H) + 2 * (self.Hr0 - self.Hr + self.H) / self.beta) * x**2 + \
                   self.Hr + self.Tr - self.Pr
        elif self.beta < x <= 1:
            Fc = (self.Emin + self.Tr - self.Pr) * x + 2 * (self.Pr - self.Tr) - self.Emin
            return 0.14 * (self.D / 2)**0.74 * (x + 1)**-3 + Fc
        else:
            return None  # Out of valid range

# Example usage
D = 100  # Crater diameter
crater = CraterHeightProfile(D)

# Define necessary parameters
H = 0.15 * D**1.01
Hr = 0.025 * D**1.01
Tr = 5
Pr = 3
Emin = 1

crater.calculate_parameters(H, Hr, Tr, Pr, Emin)

r = 30  # Example radial distance
height = crater.get_height(r)
print(f"Height at r = {r}: {height}")
