import numpy
from coopihc import ClassicControlTask, IHCT_LQGController, Bundle

I = 0.25
b = 0.2
ta = 0.03
te = 0.04

a1 = b / (ta * te * I)
a2 = 1 / (ta * te) + (1 / ta + 1 / te) * b / I
a3 = b / I + 1 / ta + 1 / te
bu = 1 / (ta * te * I)


A = numpy.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, -a1, -a2, -a3]])

B = numpy.array([[0, 0, 0, bu]]).reshape((-1, 1))

C = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])


D = numpy.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.05]])

F = numpy.diag([0.01, 0.01, 0.01, 0.01])
Y = 0.08 * B
G = 0.03 * numpy.diag([1, 0.1, 0.01, 0.001])


Q = numpy.diag([1, 0.01, 0, 0])
R = numpy.array([[1e-4]])
U = numpy.diag([1, 0.1, 0.01, 0])


D = D * 0.35
G = G * 0.35
timestep = 1e-2

task = ClassicControlTask(
    timestep, A, B, F=F, G=G, discrete_dynamics=False, noise="off"
)
user = IHCT_LQGController("user", timestep, Q, R, U, C, D, noise="on")
bundle = Bundle(task=task, user=user, start_after=1)
bundle.reset()
bundle.render("plot")
while True:
    state, rewards, flag = bundle.step()
    bundle.render("plot")
    if flag:
        break
