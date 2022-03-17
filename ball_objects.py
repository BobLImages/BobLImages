import time
import random
import math
import pygame
from colors import *


k = 300
VELOCITY_FACTOR = 6             # Changes all velocities by factor of x
RADIUS_FACTOR = 15                # if 0 then picks random from list [20,30,40] below
SOLID = 0                        # solid 0 or bordered(width)
BACKGROUND_STATUS = 2            # zone(s) pictured as 0 No Zones   1 Default color 2 based on # of balls in zone
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
        radius_list = [20,10]
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
    
    total_distance = ball_distance(ball,compare_ball)
    total_radius = ball.radius + compare_ball.radius    
    overlap = total_distance - total_radius

# Ball centers match

    if total_distance == 0:
        ball.x = ball.x - ball.xVelocity
        ball.y = ball.y - ball.yVelocity
        total_distance = ball_distance(ball,compare_ball)
        overlap = total_distance - total_radius
    
    if overlap > 0:
        return(0,overlap)

    if overlap <= 0:
        return (1,overlap)


def ts(ball,compare_ball,dbbc,e):
        
        pass
        #print("{} {} {} {} {} {} {} {} {} {}".format(e, ball.ball_id, compare_ball.ball_id, ball.y,  compare_ball.y, ball.yVelocity, compare_ball.yVelocity, ball.radius, compare_ball.radius, dbbc))
        #print(" {} {} {} {} {} {} {} {} {} {} {}".format(ball.ball_id, compare_ball.ball_id, ball.y, ball.yVelocity, ball.radius, ball.colorname, compare_ball.y, compare_ball.yVelocity, compare_ball.radius,compare_ball.colorname, dbbc))


def resolve_collision(ball,compare_ball,overlap):
    
   
    total_mass = ball.mass + compare_ball.mass
    mass1  = ball.mass
    mass2 = compare_ball.mass
    velX1 = ball.xVelocity
    velX2 = compare_ball.xVelocity
    velY1 = ball.yVelocity
    velY2 = compare_ball.yVelocity
    ball.collisions = ball.collisions + 1
    compare_ball.collisions = compare_ball.collisions + 1



   #  Fix Static collision 

    #  Non-border-ball-ball collision

    total_distance = ball_distance(ball,compare_ball)


    if ball.x > ball.radius and total_distance > 0: 
        overlap = (overlap/2)
        ball.x  = ball.x -  (overlap * (ball.x - compare_ball.x)/ total_distance)
        ball.y  = ball.y - (overlap * (ball.y - compare_ball.y)/ total_distance)

        compare_ball.x  = compare_ball.x + (overlap * (ball.x - compare_ball.x)/ total_distance)
        compare_ball.y  = compare_ball.y + (overlap * (ball.y - compare_ball.y)/ total_distance)
    else:
        compare_ball.x  = compare_ball.x + (overlap * (ball.x - compare_ball.x)/ total_distance)
        compare_ball.y  = compare_ball.y + (overlap * (ball.y - compare_ball.y)/ total_distance)

    #ts(ball, compare_ball, dbbc,''static')

   

   #    MOMENTUM CHANGE CALCULATION

  
    KE_i_x = .5 * ball.mass * ball.xVelocity**2 + .5 * compare_ball.mass * compare_ball.xVelocity**2



    newVelX1 = (velX1 * (mass1 - mass2) + (2 * mass2 * velX2)) / (total_mass)
    newVelX2 = (velX2 * (mass2 - mass1) + (2 * mass1 * velX1)) / (total_mass)
    newVelY1 = (velY1 * (mass1 - mass2) + (2 * mass2 * velY2)) / (total_mass)
    newVelY2 = (velY2 * (mass2 - mass1) + (2 * mass1 * velY1)) / (total_mass)
        
    
    ball.xVelocity = newVelX1
    compare_ball.xVelocity = newVelX2
    
    ball.yVelocity = newVelY1
    compare_ball.yVelocity = newVelY2
    
    ball.x = ball.x + newVelX1
    ball.y = ball.y + newVelY1
    compare_ball.x = compare_ball.x + newVelX2
    compare_ball.y = compare_ball.y + newVelY2
    
   
    if compare_ball.x + compare_ball.radius > width:
        compare_ball.x = width - 2 * compare_ball.radius
        compare_ball.xVelocity = -newVelX2
        
    if compare_ball.x - compare_ball.radius < 0:
        compare_ball.x = 2 * compare_ball.radius
        compare_ball.xVelocity = newVelX2
 
    if compare_ball.y + compare_ball.radius > height:
        compare_ball.y = height - 2 * compare_ball.radius
        compare_ball.yVelocity = -newVelY2

    if compare_ball.y - compare_ball.radius < 0:
        compare_ball.y = 2 * compare_ball.radius
        compare_ball.yVelocity = newVelY2
    
 
    if ball.x + ball.radius > width:
        ball.x = width - 2 * ball.radius
        ball.xVelocity = -newVelX1
        
    if ball.x - ball.radius < 0:
        ball.x = 2 * ball.radius
        ball.xVelocity = newVelX1

    if ball.y + ball.radius > height:
        ball.y = height - 2 * ball.radius
        ball.yVelocity = -newVelY1

    if ball.y - ball.radius < 0:
        ball.y =  2* ball.radius
        ball.yVelocity = newVelY1 

    ball.position = ball.x, ball.y
    compare_ball.position = compare_ball.x,compare_ball.y        

    ball.cosine = ball.xVelocity/ball.radius
    ball.sine = ball.yVelocity/ball.radius 
    ball.momentum_x  = ball.mass * ball.xVelocity
    ball.momentum_y  = ball.mass * ball.yVelocity
    ball.KE_x = .5 * ball.mass * (ball.xVelocity**2)
    ball.KE_y = .5 * ball.mass * (ball.yVelocity**2)

    compare_ball.cosine = compare_ball.xVelocity/compare_ball.radius
    compare_ball.sine = compare_ball.yVelocity/compare_ball.radius 
    compare_ball.momentum_x  = compare_ball.mass * compare_ball.xVelocity
    compare_ball.momentum_y  = compare_ball.mass * compare_ball.yVelocity
    compare_ball.KE_x = .5 * compare_ball.mass * (compare_ball.xVelocity**2)
    compare_ball.KE_y = .5 * compare_ball.mass * (compare_ball.yVelocity**2)


    #    print(KE_i_x)

    # ball_momentum_x_i = ball.mass * ball.xVelocity * ball.sine
    # print('__',ball.mass,ball.xVelocity,ball.sine)
    # compare_ball_momentum_x_i = compare_ball.mass * compare_ball.xVelocity * compare_ball.sine
    # print('__',ball. ompare_ball.sine)
    # print('__',compare_ball.mass,compare_ball.xVelocity,compare_ball.sine)


    # ball_momentum_y_i = ball.mass * ball.xVelocity * ball.cosine
    # compare_ball_momentum_y_i = compare_ball.mass * compare_ball.xVelocity * compare_ball.cosine


    # ball_KE_x_i =  .5 * ball.mass * ball.xVelocity**2
    # compare_ball_KE_x_i = .5 * compare_ball.mass * compare_ball.xVelocity**2


    # ball_KE_y_i = .5 * ball.mass * ball.xVelocity**2
    # compare_ball_KE_y_i = .5 * compare_ball.mass * compare_ball.xVelocity**2

    # print("ball x_momentum {} compare ball x_momentum {} ball x_KE {} compare ball x_KE initial {}" .format(ball.momentum_x, compare_ball.momentum_x, ball.KE_x, compare_ball.KE_x))
    # print("ball y_momentum {} compare ball momentum y initial {} ball KE y initial {} compare ball KE y initial {}" .format(ball.momentum_y, compare_ball.momentum_y, ball.KE_y, compare_ball.KE_y))
    # print('')
    # print('total x momentum {} total x KE {}'.format(ball.momentum_x + compare_ball.momentum_x,ball.KE_x + compare_ball.KE_x))
    # print('total y momentum {} total y KE {}'.format(ball.momentum_y + compare_ball.momentum_y,ball.KE_y + compare_ball.KE_y))
    # print('')
    # print('')


class Ball:
    def __init__(self,id_num,x,y,radius,xV,yV,color):

        self.ball_id = id_num
        
        self.x = x
        self.y = y
        self.position = (x,y)
        self.old_x = 0
        self.old_y = 0
        
        self.color = color
        self.colorname = get_key(color)
        
        self.radius = radius
        self.xVelocity = xV
        self.yVelocity = yV

        if self.radius < 20:
            self.mass = .75
        elif self.radius == 20:
            self.mass = 1
        elif self.radius == 25:
            self.mass = 1
        elif self.radius == 30:
            self.mass = 1.02
        elif self.radius == 40:
            self.mass = 1.025
        elif self.radius == 50:
            self.mass = 1.04
        else:
            self.mass = 1.10
    

        self.momentum_x  = self.mass * self.xVelocity 
        self.momentum_y  = self.mass * self.yVelocity
        self.collisions = 0

        self.BB = self.BBox(self.ball_id,self.x,self.y,self.radius,self.color)
        self.Label = self.Label(self.ball_id,self.x,self.y,self.radius,self.color) 
    
 
        self.direction = "N" if self.yVelocity < 0  else "S"            
        self.direction = self.direction + "W" if self.xVelocity < 0 else self.direction + "E"            


    def update_direction(self):

        self.direction = "N" if self.yVelocity < 0  else "S"            
        self.direction = self.direction + "W" if self.xVelocity < 0 else self.direction + "E"            

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
            
            
            if self.x + self.radius > width or self.x - self.radius < 0:
                self.xVelocity = - self.xVelocity

            if self.y + self.radius  > height or self.y - self.radius < 0:
               self.yVelocity = -self.yVelocity
                
              # if self.y + self.radius  > height and self.yVelocity > 0:
              #   self.yVelocity = self.yVelocity + 0
              #   if self.y + self.radius  > height and self.yVelocity < 0:
              #       self.yVelocity = self.yVelocity - 0
                    

            self.old_position = self.position
            self.old_y = self.y

            self.y  = self.y + self.yVelocity
            self.x  = self.x + self.xVelocity
            self.position = (self.x, self.y)

            self.momentum_x  = self.mass * self.xVelocity
            self.momentum_y  = self.mass * self.xVelocity





            print(self.momentum_x,self.momentum_y)
 
    class BBox:
            def __init__(self,ball_id,x,y,radius,color):
                
                left_x = x - radius
                left_y = y - radius
                
                # Make separate function
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

class Grid(Zone):

    def __init__(self,counter,left_x,left_y, w,h,color):
        super().__init__(counter,left_x,left_y, w,h,color)

def create_zones():

        rows = 5
        columns = 10

        c_x = [(x*width/rows) for x in range(rows)]
        c_y = [(x*height/columns) for x in range(columns)]
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

def create_grids():

        rows = 50
        columns = 10

        c_x = [(x*width/rows) for x in range(rows)]
        c_y = [(x*height/columns) for x in range(columns)]
        NW = [(x,y) for x in c_x for y in c_y]
        #print(NW)
        zone_width = width/len(c_x)
        zone_height = height/len(c_y)

        return [Grid(i,NW[i][0], NW[i][1],zone_width,zone_height,(0,0,0)) for i in range(len(NW))]

def create_balls(grids):

        balls_created = 0
        for i,grid in enumerate(grids):
            if balls_created < k:
                balls.append(Ball(i,grid.NW[0] + grid.zone_width/2,grid.SW[1] + grid.zone_height/2, set_radius(),
                    set_velocity(), set_velocity(), color = random.choice(list(cvr.values()))))
                balls_created = balls_created + 1
            else:
                pass

        for i, ball in enumerate(balls):        
            ball.BB.bb_id = i
        return balls














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




    # ball_list = [i for i in range(k)]
    # x = int(k/100)
    # r = [ball_list[i:i+x] for i in range(0, len(ball_list), x)]
    # random.shuffle(r)
    # s = [f for f in range(len(zones))]
    # random.shuffle(s)
    # t = {s[i]: u for i,u in enumerate(r)}
    # for key,value in t.items():
    #     for zone in zones:
    #         z = zone
    #         if key == zone.id:
    #             z = zone    
    #         new_balls = new_ball = [ball for ball in balls for i in range(int(k/100)) if value[i] == ball.ball_id]
    #         reset_ball(new_balls)




def game(SOLID,BACKGROUND_STATUS,VELOCITY_FACTOR):

    zone_balls = {}    
    grids = create_grids()
    zones = create_zones()
    balls = create_balls(grids)
    hjk= {i:random.choice(list(cvr.values())) for i in range(20)}
    
    while True:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    print("left mouse button")
                    return
                elif event.button == 3:
                    print("right mouse button")
                elif event.button == 4:
                    print("mouse wheel up")
                elif event.button == 5:
                    print("mouse wheel down")

        display.fill((255,255,255))

        if BACKGROUND_STATUS == 1:
            for zone in zones:
                zone.color = (0,0,0)
        else:
            for zone in zones:
                zone.color = (255,255,255)


        for ball in balls:
            mpx,mpy = pygame.mouse.get_pos()
            if mpx > ball.BB.NW[0] and  mpx < ball.BB.SE[0] and mpy > ball.BB.NW[1] and mpy < ball.BB.SE[1]:
                print(ball.ball_id,ball.colorname,ball.BB.zone)
                pass

        [ball.move() for ball in balls]


        
        for zone_counter in range(len(zones)):
            hj = []
            hj = [ball for ball in balls if zone_counter in ball.BB.zone]
            if len(hj) > 1:
                zone_balls.update({zone_counter:hj})                    

        for key,value in zone_balls.items():
            if BACKGROUND_STATUS == 2:
                for key,value in zone_balls.items():
                    if len(value) in(2,3,4):
                        zones[key].color = (0,255,127)
                        zones[key].filled = SOLID
                    elif len(value) > 5 and len(value) <= 10:
                        zones[key].color = (46,139,87)
                        zones[key].filled = SOLID
                    elif len(value) > 10 and len(value) <= 15:
                        zones[key].color = (255,255,0)
                        zones[key].filled = SOLID
                    elif len(value) > 15 and len(value) <= 17:
                        zones[key].color = (243,66,66)
                        zones[key].filled = SOLID
                    elif len(value) > 17:
                        zones[key].color = (255,0,0)
                        zones[key].filled = SOLID
        
        [zone.draw() for zone in zones if BACKGROUND_STATUS in (1,2)]
            #[grid.draw() for grid in grids] 
        [ball.draw() for ball in balls]

        # time.sleep(5)

        for key,value in zone_balls.items():
            if len(value) == 2:
                collided = detect_collision(value[0],value[1])
                if collided[0]:
                    resolve_collision(value[0],value[1],collided[1])
            else:
                for ball in value:
                    for ball_2 in  value:
                        if ball. ball_id != ball_2.ball_id:
                            collided = detect_collision(ball,ball_2)
                            if collided[0]:
                                resolve_collision(ball,ball_2,collided[1])
        pygame.display.update()

        for ball in balls:
            ball.BB.move(ball.x,ball.y,ball.radius)
            ball.update_direction() 
            ball.BB.zone = []
            if PRINT_COLLISIONS:
                print(ball.ball_id, ball.collisions)

        [ball.BB.zone.append(i) for ball in balls for i,zone in enumerate(zones) if rect_vs_rect(ball.BB,zone)]
            
        clock.tick(FPS)


game(SOLID,BACKGROUND_STATUS,VELOCITY_FACTOR)
pygame.quit()



