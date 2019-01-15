from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
# 画图函数
def drawFunc():
    # clear新颜色,从缓存区写入
    glClear(GL_COLOR_BUFFER_BIT)
    # 画分割线,按顺序排，2点确定一条直线
    glBegin(GL_LINES)
    glVertex2f(-1.0, 0.0)
    glVertex2f(1.0, 0.0)
    glVertex2f(0.0, 1.0)
    glVertex2f(0.0, -1.0)
    glEnd()

    # 画点的位置
    glPointSize(20.0)   # 点的大小
    glBegin(GL_POINTS)
    glColor3f(1.0, 0.0, 0.0)    #红色
    glVertex2f(0.3, 0.3)
    glColor3f(0.0, 1.0, 0.0)    #绿色
    glVertex2f(0.6, 0.6)
    glColor3f(0.0, 0.0, 1.0)    #蓝色
    glVertex2f(0.9, 0.9)
    glEnd()

    # 画正方形
    glColor3f(1.0, 1.0, 0)
    glBegin(GL_QUADS)
    # 现对于中间的位置，注意按照顺序排序4点闭合成4边形
    glVertex2f(-0.2, 0.2)
    glVertex2f(-0.2, 0.5)
    glVertex2f(-0.5, 0.5)
    glVertex2f(-0.5, 0.2)

    glVertex2f(-0.6, 0.1)
    glVertex2f(-0.6, 0.3)
    glVertex2f(-0.8, 0.2)
    glVertex2f(-0.8, 0.1)

    glEnd()

    # 画六边形
    glColor3f(0.0, 1.0, 1.0)
    glPolygonMode(GL_FRONT, GL_LINE)
    #glPolygonMode(GL_BACK, GL_FILL)
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
    #glPolygonMode(GL_BACK, GL_LINE)
    glBegin(GL_POLYGON)
    glVertex3f(0.5, -0.1, 0.1)
    glVertex3f(0.2, -0.3, 0.1)
    glVertex3f(0.2, -0.6, 0.1)
    glVertex3f(0.5, -0.8, 0.1)
    glVertex3f(0.8, -0.6, 0.1)
    glVertex3f(0.8, -0.3, 0.1)
    glEnd()

    glFlush()

n = 50
R = 0.5

def circle():
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POLYGON)
    for i in range(0, n):
        glVertex2f(R*math.cos(2 * math.pi / n*i), R*math.sin(2 * math.pi / n*i))
    glEnd()
    glFlush()
# ----初始化-----#
glutInit()
glutInitDisplayMode(GLUT_RGBA|GLUT_SINGLE)
glutInitWindowPosition(400, 200)
glutInitWindowSize(400, 400)
glutCreateWindow("Sencond")
# ----初始完毕-----#

glutDisplayFunc(circle)
# RGBA颜色设置,只set,不clear,将颜色进入缓存区
#glClearColor(0.0, 0.0, 0.0, 1.0)
# 定义裁剪面
# void gluOrtho2D(GLdouble left,GLdouble right,GLdouble bottom,GLdouble top)
#gluOrtho2D(-1.0, 1.0, -1.0, 1.0)
glutMainLoop()