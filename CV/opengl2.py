from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0)

def drawFunc():
    glClear(GL_COLOR_BUFFER_BIT)
    # 画分割线
    glBegin(GL_LINES)
    glVertex2f(-1.0, 0.0)
    glVertex2f(1.0, 0.0)
    glVertex2f(0.0, 1.0)
    glVertex2f(0.0, -1.0)
    glEnd()

    # 画点的位置
    glPointSize(20.0)
    glBegin(GL_POINTS)
    glColor3f(1.0, 0.0, 0.0)
    glVertex2f(0.3, 0.3)
    glColor3f(0.0, 1.0, 0.0)
    glVertex2f(0.6, 0.6)
    glColor3f(0.0, 0.0, 1.0)
    glVertex2f(0.9, 0.9)
    glEnd()

    # 画正方形
    glColor3f(1.0, 1.0, 0)
    glBegin(GL_QUADS)
    #相对于分割线的位置
    glVertex2f(-0.2, 0.2)
    glVertex2f(-0.2, 0.5)
    glVertex2f(-0.5, 0.5)
    glVertex2f(-0.5, 0.2)
    glEnd()

    # 画六边形
    glColor3f(0.0, 1.0, 1.0)
    glPolygonMode(GL_FRONT, GL_LINE)
    glPolygonMode(GL_BACK, GL_FILL)
    glBegin(GL_POLYGON)
    glVertex2f(-0.5, -0.1)
    glVertex2f(-0.8, -0.3)
    glVertex2f(-0.8, -0.6)
    glVertex2f(-0.5, -0.8)
    glVertex2f(-0.2, -0.6)
    glVertex2f(-0.2, -0.3)
    glEnd()

    # 画有填充颜色的六边形
    glPolygonMode(GL_FRONT, GL_FILL)
    glPolygonMode(GL_BACK, GL_LINE)
    glBegin(GL_POLYGON)
    glVertex3f(0.5, -0.1, 0.1)
    glVertex3f(0.2, -0.3, 0.1)
    glVertex3f(0.2, -0.6, 0.1)
    glVertex3f(0.5, -0.8, 0.1)
    glVertex3f(0.8, -0.6, 0.1)
    glVertex3f(0.8, -0.3, 0.1)
    glEnd()

    glFlush()

glutInit()
glutInitDisplayMode(GLUT_RGBA|GLUT_SINGLE)
glutInitWindowSize(400, 400)
glutCreateWindow("Sencond")

glutDisplayFunc(drawFunc)
init()
glutMainLoop()