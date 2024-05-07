f_text = "log(exp(z)-3+exp(-z)-z^2)"
Xrange = [-2, 2]
Yrange = [-2, 2]
checkCReqs = False

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# make it symbolic
f_z = sym.sympify(f_text)

# replace f(z) by f(x+iy)
x, y = sym.symbols("x y", real=True)
f_xy = f_z.subs("z", x + y * 1j)
f_xy = f_xy.subs("i", 1j)
fNumpy = sym.lambdify((x, y), f_xy, "numpy")

# check Cauchy Riemann Equations
if checkCReqs:
    u = sym.re(f_xy)
    v = sym.im(f_xy)
    ux = u.diff(x)
    vx = v.diff(x)
    uy = u.diff(y)
    vy = v.diff(y)
    uxEQvy = (ux - vy).simplify() == 0
    uyEQnvx = (uy + vx).simplify() == 0
    if uxEQvy and uyEQnvx:
        print("Cauchy-Riemann Equations satisfied")
    else:
        raise Warning("Cauchy-Riemann Equations not satisfied")

# mpl setup
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.75, "image.cmap": "jet"})


X, Y = np.meshgrid(np.linspace(*[*Xrange, 250]), np.linspace(*[*Yrange, 250]))
fPts = fNumpy(X, Y)
theta = np.linspace(0, 2 * np.pi, int(1e4))
fUnitCirc = fNumpy(np.cos(theta), np.sin(theta))
linScale = np.linspace(-100, 100, int(1e4))
fReAx = fNumpy(linScale, 0)
fImAx = fNumpy(linScale, 0)

## Real and Imaginary Parts
plt.figure(figsize=(10, 4))
plt.suptitle("$f(z)=" + sym.latex(f_z) + "$")
plt.subplot(1, 2, 1)
plt.scatter(X, Y, c=np.real(fPts))
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$\Re f(z)$")
plt.axis([*Xrange, *Yrange])
plt.colorbar()
plt.subplot(1, 2, 2)
plt.scatter(X, Y, c=np.imag(fPts))
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$\Im f(z)$")
plt.axis([*Xrange, *Yrange])
plt.colorbar()
plt.tight_layout()


## Angle and Magnitude
plt.figure(figsize=(10, 4))
plt.suptitle("$f(z)=" + sym.latex(f_z) + "$")
plt.subplot(1, 2, 1)
plt.scatter(X, Y, c=np.abs(fPts))
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$|f(z)|$")
plt.axis([*Xrange, *Yrange])
plt.colorbar()
plt.subplot(1, 2, 2)
plt.scatter(X, Y, c=np.angle(fPts))
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$\\angle f(z)$")
plt.axis([*Xrange, *Yrange])
plt.colorbar()
plt.tight_layout()


plt.figure(figsize=(12, 8))
plt.set_cmap("hsv")
plt.suptitle("$f(z)=" + sym.latex(f_z) + "$")

plt.subplot(2, 3, 1)
plt.scatter(np.cos(theta), np.sin(theta), c=theta)
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$z$")

plt.subplot(2, 3, 1 + 3)
plt.scatter(np.real(fUnitCirc), np.imag(fUnitCirc), c=theta)
plt.xlabel("$\Re f(z)$")
plt.ylabel("$\Im f(z)$")
plt.title("$f(z)$")

plt.subplot(2, 3, 2)
plt.scatter(linScale, 0 * linScale, c=linScale)
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$z$")

plt.subplot(2, 3, 2 + 3)
plt.scatter(np.real(fReAx), np.imag(fReAx), c=linScale)
plt.xlabel("$\Re f(z)$")
plt.ylabel("$\Im f(z)$")
plt.title("$f(z)$")

plt.subplot(2, 3, 3)
plt.scatter(0 * linScale, linScale, c=linScale)
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$z$")

plt.subplot(2, 3, 3 + 3)
plt.scatter(np.real(fImAx), np.imag(fImAx), c=linScale)
plt.xlabel("$\Re f(z)$")
plt.ylabel("$\Im f(z)$")
plt.title("$f(z)$")

plt.tight_layout()
plt.show()
