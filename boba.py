   
import pygame
import pymunk
import random

pygame.init()

display = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
space = pymunk.Space()
FPS = 50

def convert_coordinates(point):
    return int(point[0]), 600-int(point[1])

class Ball():
    def __init__(self, x, y, collision_type, up = 1):
        self.body = pymunk.Body()
        self.body.position = x, y
        self.body.velocity = random.uniform(-100, 100), random.uniform(-100, 100)
        # self.body.velocity = 0, up*100
        self.shape = pymunk.Circle(self.body, 10)
        self.shape.elasticity = 1
        self.shape.density = 1
        self.shape.collision_type = collision_type
        space.add(self.body, self.shape)
    def draw(self):
        if self.shape.collision_type != 2:
            pygame.draw.circle(display, (255, 0, 0), convert_coordinates(self.body.position), 10)
        else:
            pygame.draw.circle(display, (0, 0, 255), convert_coordinates(self.body.position), 10)
    def change_to_blue(self, arbiter, space, data):
        self.shape.collision_type = 2

class Platform():
    def __init__(self,y, color):
        
        self.color = color
        self.y = y
        self.body = pymunk.Body(body_type =pymunk.Body.STATIC)
        self.body.position = 0,y
        self.shape = pymunk.Segment(self.body,[0,0],[600,0],7)
        self.shape.elasticity = 1
        self.density = 1
        space.add(self.body, self.shape)

    def draw(self):
            a = convert_coordinates(self.body.local_to_world(self.shape.a))
            b = convert_coordinates(self.body.local_to_world(self.shape.b))
            pygame.draw.line(display,self.color, a,b,7)


class Wall():

    def __init__(self,x, color):
        
        self.color = color
        self.x= x
        self.body = pymunk.Body(body_type =pymunk.Body.STATIC)
        self.body.position = x,0
        self.shape = pymunk.Segment(self.body,[0,600],[0,0],7)
        self.shape.elasticity = 1
        self.density = 1
        space.add(self.body, self.shape)

    def draw(self):
            a = convert_coordinates(self.body.local_to_world(self.shape.a))
            b = convert_coordinates(self.body.local_to_world(self.shape.b))
            pygame.draw.line(display,self.color, a,b,7)




def game():
    
    balls = [Ball(random.randint(0, 600), random.randint(0, 600), i+3) for i in range(100)]
    balls.append(Ball(400, 400, 2))
    
    handlers = [space.add_collision_handler(2, i+3) for i in range(100)]
    for i, handler in enumerate(handlers):
        handler.separate = balls[i].change_to_blue

    platform_3= Platform(1,(0,255,0))
    platform_4 = Platform(599,(0,255,0))
    wall_1 = Wall(1,(0,255,0))
    wall_2 = Wall(599,(0,255,0))




 

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                balls.append(Ball(450,450,1))
                handlers.append(space.add_collision_handler(2, i+3))
                handlers[-1].begin =balls[-1].change_to_blue
                for i, handler in enumerate(handlers):
                    handler.separate = balls[-i].change_to_blue

        display.fill((255, 255, 255))
        [ball.draw() for ball in balls]
        platform_3.draw()
        platform_4.draw()
        wall_1.draw()
        wall_2.draw()


        pygame.display.update()
        clock.tick(FPS)
        space.step(1/FPS)

game()
pygame.quit()







