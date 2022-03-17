import time
import random
import math
import pygame
from colors import *





k = 200
VELOCITY_FACTOR = 7                    # Changes all velocities by factor of x
RADIUS_FACTOR = 20                  # if 0 then picks random from list [20,30,40] below
SOLID = 0                          # solid 0 or bordered(width)
BACKGROUND_STATUS = 0            # zone(s) pictured as 0 No Zones   1 Default color 2 based on # of balls in zone
LABELS = False                      # Turn numeric ball_id label On/Off Should be >= 20 radius to show up well  (slows program down)
PRINT_COLLISIONS = False            # Prints # collision for each ball
width = 2500
height = 1300

cvr = get_colors()


balls = []
zones = []
a = []


pygame.init()
pygame.font.init()

display = pygame.display.set_mode((width,height))
clock = pygame.time.Clock()
FPS = 60


def set_radius():
    

    if RADIUS_FACTOR:
        radius_list = [RADIUS_FACTOR]
    else:
        radius_list = [20,30,40]
    radius = random.choice(radius_list)
    
    return radius

def set_velocity():
    velocity_list = [-.5,-.4,-.3,-.2,-.1,.1,.2,.3,.4,.5]
    #velocity_list = [-.3,-.2,-.1,.1,.2,.3]
    #velocity_list = [-.1,.1]
    vel = random.choice(velocity_list) * VELOCITY_FACTOR
    return vel

def g_font(rad):

   if rad > 20:
        myfont = pygame.font.SysFont("comicsansms", 25)
   else:
        myfont = pygame.font.SysFont("comicsansms", 20)

   return myfont

def convert_to_complement(color):
    return (255 -color[0], 255-color[1], 255 - color[2])

def convert_to_contrast(color):
    j = (255 - color[0], 255 - color[1], 255 - color[2])
    
    if j[0] > 80 and j[0] < 90:
        j = (0, 255 - color[1], 255 - color[2])
    
    return j

def get_key(g):
    for key,value in cvr.items():
        if g == value:
            return key

def set_direction(ball):

    if ball.yVelocity < 0:
        ball.direction = "N"
    else:
        ball.direction = "S"            

    if ball.xVelocity < 0:
        ball.direction = ball.direction + "W"
    else:
        ball.direction = ball.direction + "E"            
 

def distance(x1,y1,x2,y2):
    
    xDistance = x2-x1
    yDistance = y2-y1
    return (xDistance**2 + yDistance**2)**.5


def ball_distance(ball_1,ball_2):
    xDistance = ball_2.x- ball_1.x
    yDistance = ball_2.y-ball_1.y
    return (xDistance**2 + yDistance**2)**.5


def ball_distance_3(ball_1,ball_2,ball_3):
    r = False
    xDistance = ball_2.x- ball_1.x
    yDistance = ball_2.y-ball_1.y
    if (xDistance**2 + yDistance**2)**.5 < 10:
        r = True


    return (xDistance**2 + yDistance**2)**.5

    xDistance = ball_3.x- ball_1.x
    yDistance = ball_3.y-ball_1.y
    if (xDistance**2 + yDistance**2)**.5 < 10:
        r = True

    xDistance = ball_3.x- ball_2.x
    yDistance = ball_3.y-ball_2.y
    if (xDistance**2 + yDistance**2)**.5 < 10:
        r = True
    return r


def pnt_vs_rect(pos,zone):
    if pos[0] >= zone.NW[0]  and pos[1] >= zone.NW[1] and pos[0] <= zone.SE[0] and pos[1] <= zone.SE[1]:
        return True
    else:
        return False         


def rect_vs_rect(bb,zone):
    #print(bb.bb_id,bb.upper_left,bb.lower_right,bb.lower_left,bb.upper_right)
    if pnt_vs_rect(bb.NW,zone) or pnt_vs_rect(bb.SE,zone) or pnt_vs_rect(bb.SW,zone) or pnt_vs_rect(bb.NE,zone):
        #print('true')
        return True
    else:
        return False         


def detect_collision(ball,compare_ball):
    
    k = 0
    
    dbbc = distance(ball.x,ball.y,compare_ball.x,compare_ball.y)
    if dbbc == 0:
        ball.x = ball.x -ball.radius
        ball.y = ball.y + ball.radius
        old_color = ball.color
        ball.color = ((255,255,255))
        ball.colorname = "White"
        old_color_2 = compare_ball.color

        compare_ball.color = (255,255,255)
        compare_ball.colorname = "White"
        if old_color != (255,255,255) or old_color_2 != (255,255,255):
            bing = pygame.mixer.Sound("C:/Users/User/Downloads/phasesr2.wav") 
            pygame.mixer.Sound.play(bing)

        dbbc = distance(ball.x,ball.y,compare_ball.x,compare_ball.y)

    total_radius = ball.radius + compare_ball.radius
    overlap = dbbc - total_radius
    if dbbc < total_radius:
        return (1,overlap)
    else:
        return(0,overlap)


def ts(ball,compare_ball,dbbc,e):
        
        pass
        #print("{} {} {} {} {} {} {} {} {} {}".format(e, ball.ball_id, compare_ball.ball_id, ball.y,  compare_ball.y, ball.yVelocity, compare_ball.yVelocity, ball.radius, compare_ball.radius, dbbc))
        #print(" {} {} {} {} {} {} {} {} {} {} {}".format(ball.ball_id, compare_ball.ball_id, ball.y, ball.yVelocity, ball.radius, ball.colorname, compare_ball.y, compare_ball.yVelocity, compare_ball.radius,compare_ball.colorname, dbbc))


def resolve_collision(ball,compare_ball,overlap):
    
   #  Fix Static collision 

    #  Non-border-ball-ball collision

    
    ball.collisions = ball.collisions + 1
    compare_ball.collisions = compare_ball.collisions + 1

    

    dbbc  = distance(ball.x,ball.y,compare_ball.x,compare_ball.y)


    t = ball.y
    s = compare_ball.y
    #ts(ball, compare_ball, dbbc,'START')


    if ball.x > ball.radius and dbbc > 0: 
        overlap = (overlap/2)
        ball.x  = ball.x -  (overlap * (ball.x - compare_ball.x)/ dbbc)
        ball.y  = ball.y - (overlap * (ball.y - compare_ball.y)/ dbbc)

        compare_ball.x  = compare_ball.x + (overlap * (ball.x - compare_ball.x)/ dbbc)
        compare_ball.y  = compare_ball.y + (overlap * (ball.y - compare_ball.y)/ dbbc)
    else:
        compare_ball.x  = compare_ball.x + (overlap * (ball.x - compare_ball.x)/ dbbc)
        compare_ball.y  = compare_ball.y + (overlap * (ball.y - compare_ball.y)/ dbbc)

    #ts(ball, compare_ball, dbbc,''static')

    mass1  = ball.mass
    mass2 = compare_ball.mass
    velX1 = ball.xVelocity
    velX2 = compare_ball.xVelocity
    velY1 = ball.yVelocity
    velY2 = compare_ball.yVelocity
    
    

   #    MOMENTUM CHANGE CALCULATION


    newVelX1 = (velX1 * (mass1 - mass2) + (2 * mass2 * velX2)) / (mass1 + mass2)
    newVelX2 = (velX2 * (mass2 - mass1) + (2 * mass1 * velX1)) / (mass1 + mass2)
    newVelY1 = (velY1 * (mass1 - mass2) + (2 * mass2 * velY2)) / (mass1 + mass2)
    newVelY2 = (velY2 * (mass2 - mass1) + (2 * mass1 * velY1)) / (mass1 + mass2)
        
    









    ball.xVelocity = newVelX1
    compare_ball.xVelocity = newVelX2
    ball.yVelocity = newVelY1
    compare_ball.yVelocity = newVelY2
    
    ball.x = ball.x + newVelX1
    ball.y = ball.y + newVelY1
    
    #ts(ball, compare_ball, dbbc,''physics for ball')
   


    compare_ball.x = compare_ball.x + newVelX2
    compare_ball.y = compare_ball.y + newVelY2
 
    if compare_ball.x + compare_ball.radius > width:
        compare_ball.x = width - compare_ball.radius
        compare_ball.xVelocity = -newVelX2
        
    if compare_ball.x - compare_ball.radius < 0:
        compare_ball.x = compare_ball.radius
        compare_ball.xVelocity = newVelX2
 
    if compare_ball.y + compare_ball.radius > height:
        compare_ball.y = height - compare_ball.radius
        compare_ball.yVelocity = -newVelY2

    if compare_ball.y - compare_ball.radius < 0:
        compare_ball.y = compare_ball.radius
        compare_ball.yVelocity = newVelY2
    
 
    if ball.x + ball.radius > width:
        ball.x = width - ball.radius
        ball.xVelocity = -newVelX1
        
    if ball.x - ball.radius < 0:
        ball.x = ball.radius
        ball.xVelocity = newVelX1

    if ball.y + ball.radius > height:
        ball.y = height - ball.radius
        ball.yVelocity = -newVelY1

    if ball.y - ball.radius < 0:
        ball.y = ball.radius
        ball.yVelocity = newVelY1 

    ball.position = ball.x, ball.y
    compare_ball.position = compare_ball.x,compare_ball.y        


class Ball:
    def __init__(self,id_num,x,y,radius,xV,yV,color):

        self.ball_id = id_num
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.xVelocity = xV
        self.yVelocity = yV
        self.position = (x,y)
        self.colorname = get_key(color)
        self.old_x = 0
        self.old_y = 0
        self.collisions = 0 
        self.direction = None
        self.team = None


        self.BB = self.BBox(self.ball_id,self.x,self.y,self.radius,self.color)
        self.Label = self.Label(self.ball_id,self.x,self.y,self.radius,self.color) 
    
 

        if self.yVelocity < 0:
            self.direction = "N"
        else:
            self.direction = "S"            

        if self.xVelocity < 0:
            self.direction = self.direction + "W"
        else:
            self.direction = self.direction + "E"            


        if self.radius < 20:
            self.mass = .95
        if self.radius == 20:
            self.mass = 1
        if self.radius == 25:
            self.mass = 1
        if self.radius == 30:
            self.mass = 1.02
        if self.radius == 40:
            self.mass = 1.025
        if self.radius == 50:
            self.mass = 1.04



    def update_direction(self):          

        if self.yVelocity < 0:
            self.direction = "N"
        else:
            self.direction = "S"            

        if self.xVelocity < 0:
            self.direction = self.direction + "W"
        else:
            self.direction = self.direction + "E"            

    def draw(self):

        pygame.draw.circle(display, self.color, (self.x,self.y), self.radius)
        
        if LABELS and self.color != (255,255,255):
            font =  g_font(self.radius)
            label = font.render(self.Label.str_ball, 1, self.Label.contrast)
            if self.Label.label_id > 99:
                display.blit(label, (self.x - 20,self.y-15)) 
            else:
                display.blit(label, (self.x - 15,self.y-15)) 




    def move(self):
            
            self.old_position = self.position
            self.old_y = self.y
            
            if self.x + self.radius > width or self.x - self.radius < 0:
                self.xVelocity = - self.xVelocity
            if self.x + self.radius > width:
                bing = pygame.mixer.Sound("C:/Users/User/Downloads/phasesr2.wav") 
                # pygame.mixer.Sound.play(bing)
                # self.color = (0,0,0)
                # self.colorname = 'black' 
            else:
                pass 

            if self.y + self.radius  > height or self.y - self.radius < 0:
                
                self.yVelocity = -self.yVelocity
                if self.y + self.radius  > height and self.yVelocity > 0:
                    self.yVelocity = self.yVelocity + 0
                if self.y + self.radius  > height and self.yVelocity < 0:
                    self.yVelocity = self.yVelocity - 0
                    

            if self.y + self.radius  > height:
                bing = pygame.mixer.Sound("C:/Users/User/Downloads/phasesr2.wav") 
                # pygame.mixer.Sound.play(bing) 
                # self.color = (0,0,0)
                # self.colorname = 'black' 

            self.y  = self.y + self.yVelocity
            self.x  = self.x + self.xVelocity
            self.position = (self.x, self.y)


 
    class BBox:
            def __init__(self,ball_id,x,y,radius,color):
                left_x = x - radius
                left_y = y - radius
                color_brightness =  (color[0] +color[1] + color[2])/3
                color = (color_brightness,color_brightness,color_brightness)

                self.bb_id  = ball_id
                self.color = color
                self.NW = (left_x,left_y)
                self.SE = (left_x + radius*2,left_y + radius*2) 
                self.NE = (left_x + radius*2, left_y) 
                self.SW = (left_x, left_y + radius*2)
                self.w = radius*2
                self.h = radius*2 
                self.zone = []

            

            def move(self,x,y,radius):
                
                self.NW = (x-radius,y-radius)
                self.SE = (x + radius*2,y + radius*2) 
                self.NE = (x + radius*2, y) 
                self.SW = (x, y + radius*2)

                #pygame.draw.rect(display,self.color,(int(self.NW[0]),int(self.NW[1])),(int(self.SE[0]),int(self.SE[1])),3)


            def draw(self):
                        
                pygame.draw.rect(display,self.color,int(self.NW[0]),int(self.NW[1]),int(self.SE[0]),int(self.SE[1]),3)


    class Label:
            def __init__(self,ball_id,x,y,radius,color):

                self.label_id = ball_id
                self.str_ball = str(self.label_id)
                self.radius = radius
                self.x = x
                self.y = y
                self.color = color
                self.complement = convert_to_complement(color)
                self.compliment_name = get_key(self.complement)
                self.contrast = convert_to_contrast(color)
                self.contrast_name = get_key(self.contrast)


                self.font =  g_font(radius)
                self.label = self.font.render(self.str_ball, 1, self.complement)

            def draw(self):

                if self.label_id > 99:
                    display.blit(self.label, (self.x - 20,self.y-15)) 
                else:
                    display.blit(self.label, (self.x - 15,self.y-15)) 

            def move(self):

                    self.x = self.x - 20
                    self.y = self.y - 15 

class Zone():

    def __init__(self,counter,left_x,left_y, w,h,color):

        self.NW = (left_x,left_y)
        self.SE = (left_x + w,left_y + h) 
        self.SW = (left_x,left_y + h)
        self.NE = (left_x+ w + w,left_y) 
        self.color = color
        
        self.zone_width = w
        self.zone_height = h
        self.id = counter
        self.filled = 2
        self.zone_members = []


    def draw(self):
            
        pygame.draw.rect(display,self.color,(int(self.NW[0]),int(self.NW[1]),self.zone_width,self.zone_height),self.filled)

#pygame.draw.line(screen, Color_line, (60, 80), (130, 100))

class Line():

    def __init__(self,counter,x,y,x1,y1,color = (0,0,0)):

        self.start_point = (x, y)
        self.end_point = (x1, y1) 
        self.color = color
        self.line_id = counter


    def draw(self):
            
        pygame.draw.line(display, self.color, (self.start_point[0],self.start_point[1]),(self.end_point[0],self.end_point[1]),2)









def create_zones():

        c_x = [(x*250) for x in range(10)]
        c_y = [(x*130) for x in range(10)]
        NW = [(x,y) for x in c_x for y in c_y]
        #print(NW)
        zone_width = width/len(c_x)
        zone_height = height/len(c_y)

        return [Zone(i,NW[i][0], NW[i][1],zone_width,zone_height,(0,0,0)) for i in range(len(NW))]


def reset_ball(new_balls):

    for zone in zones:
        offset = 12
        for ball in new_balls:
            ball.x = zone.NW[1] + offset
            ball.y = zone.NW[1] + offset
            offset = offset + 12
            ball.BB.move(ball.x,ball.y,ball.radius)


def create_balls(zones):

    balls = [Ball(i,random.randint(10 + 5,width-(10 +5)), random.randint(10 + 5, height-(10 + 5)), 
            set_radius(),set_velocity(), set_velocity(), color = random.choice(list(cvr.values()))) for i in range(k)]



    # ball_list = [ball for ball in balls if ball.ball_id < int(k/2)]
    # a = 0
    # b = 75
    # c = 0
    # d = 39
    # for ball in ball_list:
    #     ball.x = (a + b)
    #     ball.y = (c +  d )
    #     a = a + b
    #     b = b + b
    #     if a > 1250:
    #         a = 0
    #         b = 75
    #         c = d
    #         d = d + 39
    #         ball.BB.move(ball.x,ball.y,ball.radius)

    


    # ball_list = [ball for ball in balls if ball.ball_id >= int(k/2)]
    # a = 2500
    # b = -75
    # c = 0
    # d = 39
    # for ball in ball_list:
    #     ball.x = (a + b) 
    #     ball.y =  (c +  d )
    #     print(ball.x,ball.y)
    #     a = a + b
    #     b = b + b
    #     if a < 1250:
    #         a = 2500
    #         b = -75
    #         c = d
    #         d = d + 39
    #         ball.BB.move(ball.x,ball.y,ball.radius)




    ball_list = [i for i in range(k)]
    x = int(k/100)
    r = [ball_list[i:i+x] for i in range(0, len(ball_list), x)]
    random.shuffle(r)
    s = [f for f in range(len(zones))]
    random.shuffle(s)
    t = {s[i]: u for i,u in enumerate(r)}
    for key,value in t.items():
        for zone in zones:
            z = zone
            if key == zone.id:
                z = zone    
            new_balls = new_ball = [ball for ball in balls for i in range(int(k/100)) if value[i] == ball.ball_id]
            reset_ball(new_balls)

    for i, ball in enumerate(balls):        
        ball.BB.bb_id = i
        # if ball.x < 1250:
        #     ball.xVelocity = abs(ball.xVelocity) + 3
        #     ball.yVelocity = abs(ball.yVelocity)
        #     ball.color = (255,0,0)
        #     ball.colorame = "Red"
        # else:
        #     ball.xVelocity = 0
        #     ball.yVelocity = 0
        print(ball.ball_id, ball.colorname)
        #     ball.colorame = "Black"

    return balls




def game():
    balls = []
    x_s = 310
    y_s = 310
    degrees = []
    lines = []

    c = 0

    while c < 360:
        degrees.append(c*math.pi/180)
        c = c + 7.50



    t1 = Ball(0,1250,650,200,0,0, (100,28,190))
    balls .append(t1)

    for i in range(len(degrees)):
        angle = degrees[i]
        x = 600 * math.cos(angle)
        y = 600 * math.sin(angle)
        t = Ball(i+1,x + 1250,y + 650,20,0,0,color = random.choice(list(cvr.values())))
        balls.append(t)

    
    for ball in balls:
        if ball.ball_id > 0:
            lines.append(Line(ball.ball_id,ball.x, ball.y,1250,650,(0,0,0)))




    out = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                return
        display.fill((255,255,255))
        balls[0].draw()
        [line.draw() for line in lines]
        for j in range(1,len(balls)):
            balls[j].draw() 

        pygame.display.update()
        clock.tick(FPS)



        #time.sleep(4)
        if out:
            x_s = x_s +.8
            y_s = y_s + .8
            for i in range(len(degrees)):
                angle = degrees[i] + .005
                degrees[i] = angle
        else:
            x_s = x_s - .8
            y_s = y_s - .8
            for i in range(len(degrees)):
                angle = degrees[i] - .005
                degrees[i] = angle
                  

        for ball in balls:
            if ball.ball_id > 0:
                x = x_s * math.cos(degrees[ball.ball_id -1])
                y = y_s * math.sin(degrees[ball.ball_id -1])
                print(ball.ball_id,x,y)
                ball.x = x + 1250
                ball.y = y + 650

        


        if ball_distance(balls[1],balls[2]) > 4 and out:
            for i in range(1,len(balls)):
                balls[i].radius = balls[i].radius + .05
 
        if ball_distance(balls[1],balls[2]) > 4 and not out:
            for i in  range(1,len(balls)):
                balls[i].radius = balls[i].radius - .05

        if ball_distance(balls[1],balls[2]) < 4:           
            for i in  range(1,len(balls)):
                balls[i].radius = 20

        lines = []
        for ball in balls:
            if ball.ball_id > 0:
                lines.append(Line(ball.ball_id,ball.x, ball.y,1250,650,(0,0,0)))




                # if ball.ball_id < 48:
                #     lines.append(Line(ball.ball_id,ball.x, ball.y,balls[ball.ball_id + 1].x,balls[ball.ball_id + 1].y,(0,0,0)))
                # else:
                #     lines.append(Line(ball.ball_id,ball.x, ball.y,balls[1].x,balls[1].y,(0,0,0)))



        if x_s > 615:
            out = False
        if x_s < 50:
            time.sleep(2)
            out = True





     

game()
pygame.quit()



