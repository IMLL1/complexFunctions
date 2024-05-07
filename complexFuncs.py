f_text = "exp(z)+z"  # enter function
Xrange = [-10, 10]  # enter plotting x and y domains
Yrange = [-10, 10]
checkCReqs = True  # whether to check Cauchy-Riemann 

## TODO:
# - Improve pole/large dynamic range detection
# - enable detection on Re/Im parts graph

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

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
plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.alpha": 0.75,
        "image.cmap": "gist_rainbow",
        "lines.markersize": 1,
    }
)

# function viz
X, Y = np.meshgrid(np.linspace(*[*Xrange, 250]), np.linspace(*[*Yrange, 250]))
fPts = fNumpy(X, Y)

# unit circle
theta = np.linspace(0, 2 * np.pi, int(1e4))
fUnitCirc = fNumpy(np.cos(theta), np.sin(theta))

# real and imag axes
linScale = np.linspace(-100, 100, int(1e4))
fReAx = fNumpy(linScale, 0)
fImAx = fNumpy(0, linScale)

# square
spacedFour = np.linspace(0, 4, int(1e4))
sideNum = np.floor(spacedFour)
spacedFour = 2 * np.mod(spacedFour, 1) - 1
zUnitSquare = (
    (sideNum == 0) * (1 + spacedFour * 1j)
    + (sideNum == 1) * (1j - spacedFour)
    + (sideNum == 2) * (-1 - 1j * spacedFour)
    + (sideNum == 3) * (-1j + spacedFour)
)
fUnitSquare = fNumpy(np.real(zUnitSquare), np.imag(zUnitSquare))

## Real and Imaginary Parts
plt.figure(figsize=(12, 5))
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
plt.figure(figsize=(12, 5))
plt.suptitle("$f(z)=" + sym.latex(f_z) + "$")
plt.subplot(1, 2, 1)
if np.mean(np.abs(fPts)) > 100 * np.median(np.abs(fPts)):
    plt.scatter(X, Y, c=np.abs(fPts), norm=col.LogNorm())
else:
    plt.scatter(X, Y, c=np.abs(fPts))
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$|f(z)|$")
plt.axis([*Xrange, *Yrange])
plt.colorbar()
plt.subplot(1, 2, 2)
plt.scatter(X, Y, c=np.angle(fPts), cmap="hsv")
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$\\angle f(z)$")
plt.axis([*Xrange, *Yrange])
plt.colorbar()
plt.tight_layout()


# mapping
plt.figure(figsize=(16, 8))
plt.suptitle("$f(z)=" + sym.latex(f_z) + "$")

plt.subplot(2, 4, 1)
plt.scatter(np.cos(theta), np.sin(theta), c=theta, cmap="hsv")
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$z$")

plt.subplot(2, 4, 1 + 4)
plt.scatter(np.real(fUnitCirc), np.imag(fUnitCirc), c=theta, cmap="hsv")
plt.xlabel("$\Re f(z)$")
plt.ylabel("$\Im f(z)$")
plt.title("$f(z)$")

plt.subplot(2, 4, 2)
plt.scatter(np.real(zUnitSquare), np.imag(zUnitSquare), c=linScale, cmap="hsv")
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$z$")

plt.subplot(2, 4, 2 + 4)
plt.scatter(np.real(fUnitSquare), np.imag(fUnitSquare), c=linScale, cmap="hsv")
plt.xlabel("$\Re f(z)$")
plt.ylabel("$\Im f(z)$")
plt.title("$f(z)$")

plt.subplot(2, 4, 4)
plt.scatter(linScale, 0 * linScale, c=linScale, cmap="hsv")
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$z$")

plt.subplot(2, 4, 4 + 4)
plt.scatter(np.real(fReAx), np.imag(fReAx), c=linScale, cmap="hsv")
plt.xlabel("$\Re f(z)$")
plt.ylabel("$\Im f(z)$")
plt.title("$f(z)$")

plt.subplot(2, 4, 3)
plt.scatter(0 * linScale, linScale, c=linScale, cmap="hsv")
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$z$")

plt.subplot(2, 4, 3 + 4)
plt.scatter(np.real(fImAx), np.imag(fImAx), c=linScale, cmap="hsv")
plt.xlabel("$\Re f(z)$")
plt.ylabel("$\Im f(z)$")
plt.title("$f(z)$")

plt.subplot(2, 4, 3)
plt.scatter(0 * linScale, linScale, c=linScale, cmap="hsv")
plt.xlabel("$\Re z$")
plt.ylabel("$\Im z$")
plt.title("$z$")

plt.subplot(2, 4, 3 + 4)
plt.scatter(np.real(fImAx), np.imag(fImAx), c=linScale, cmap="hsv")
plt.xlabel("$\Re f(z)$")
plt.ylabel("$\Im f(z)$")
plt.title("$f(z)$")

plt.tight_layout()

plt.show()
