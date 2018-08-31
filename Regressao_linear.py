#coding: utf-8
from numpy import *

def compute_SQR(b, m, points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (m*x + b))**2

    return totalError / float(len(points))

def step_gradient(current_b, current_m, points):
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -(2/n) * (y - ((current_m*x) + current_b))
        m_gradient += -(2/n) * x*(y - ((current_m * x) + current_b))
    return [b_gradient,m_gradient]

def gradient_descent_runner(points, initial_b, initial_m, learning_rate, gradient_tolerance):
    b = initial_b
    m = initial_m
    b_gradient, m_gradient = gradient_tolerance +1

    while b_gradient > gradient_tolerance and m_gradient > gradient_tolerance:
        b_gradient, m_gradient = step_gradient(b, m, array(points))
        b = b - (learning_rate * b_gradient)
        m = m - (learning_rate * m_gradient)
        print "Iteração: {0}, w0 = {1}, w1 = {2}, RSS = {3}".format((i + 1), b, m,compute_SQR(b, m, points))
    return [b,m]

def estimativa_coeficientes(points):
    n = float(len(points))
    mediaX = 0
    mediaY = 0
    for i in range(len(points)):
        mediaX += points[i,0]
        mediaY += points[i, 1]
    mediaX = mediaX / n
    mediaY = mediaY / n

    a = 0
    b = 0
    for i in range(len(points)):
        a += (points[i,0] - mediaX)*(points[i,1]-mediaY)
        b += (points[i, 0] - mediaX) ** 2

    w1 = a/b
    w0 = mediaY - w1*mediaX
    return [w0,w1]

def run():
    points = genfromtxt('income.csv', delimiter=',')
    learning_rate = 0.0025
    initial_b = 0
    initial_m = 0
    num_iterations = 20000

    [b,m] = gradient_descent_runner(points,initial_b,initial_m,learning_rate, num_iterations)


if __name__ == '__main__':
    run()