import pygame

class Button():
    def __init__(self, x, y, image, scale):
        width = image.get_width()
        height = image.get_height()
        self.image = pygame.transform.scale(image, (int(width * scale), int(height * scale)))
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)

    def draw(self, surface):
        action = False
        # Get mouse position
        pos = pygame.mouse.get_pos()

        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                self.clicked = True
                action = True

        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        surface.blit(self.image, (self.rect.x, self.rect.y))

        return action

def init():
    global screen
    pygame.init()
    # Initialing RGB Color 
    screen = pygame.display.set_mode((600, 500))

    # Changing surface color
    color = (255, 255, 255)
    screen.fill(color)
    pygame.display.flip()

    # Set caption
    pygame.display.set_caption('Donatello Tracking Drone')

    # Buttons
    launch_img = pygame.image.load('launch.png').convert_alpha()
    land_img = pygame.image.load('land.png').convert_alpha()
    right_img = pygame.image.load('right.png').convert_alpha()
    left_img = pygame.image.load('left.png').convert_alpha()
    take_photo_img = pygame.image.load('camera.png').convert_alpha()
    up_img = pygame.image.load('up.png').convert_alpha()
    down_img = pygame.image.load('down.png').convert_alpha()
    yaw_right_img = pygame.image.load('yaw-right.png').convert_alpha()
    yaw_left_img = pygame.image.load('yaw-left.png').convert_alpha()

    # Instances
    global launch_button
    global land_button
    global right_button
    global left_button
    global take_photo_button
    global up_button
    global down_button
    global yaw_right_button
    global yaw_left_button

    launch_button = Button(0, 330, launch_img, 0.7)
    land_button = Button(370, 330, land_img, 0.7)
    right_button = Button(450, 180, right_img, 0.8)
    left_button = Button(50,180, left_img, 0.8)
    take_photo_button = Button(230, 170, take_photo_img, 0.7)
    up_button = Button(250, 0, up_img, 0.8)
    down_button = Button(250, 350, down_img, 0.8)
    yaw_right_button = Button(440, 10, yaw_right_img, 0.8)
    yaw_left_button = Button(50, 10, yaw_left_img, 0.8)

def getKey(keyName):
    ans = False
    for eve in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    if keyInput[myKey]:
        ans = True
    pygame.display.update()

    return ans

def getButtonPress(keyName):
    action = ''
    for eve in pygame.event.get(): pass
    if launch_button.draw(screen):
        print('LAUNCH')
        action = 'LAUNCH'
    elif land_button.draw(screen):
        print('LAND')
        action = 'LAND'
    elif right_button.draw(screen):
        print('RIGHT')
        action = 'RIGHT'
    elif left_button.draw(screen):
        print('LEFT')
        action = 'LEFT'
    elif take_photo_button.draw(screen):
        print('PHOTO')
        action = 'PHOTO'
    elif up_button.draw(screen):
        print('UP')
        action = 'UP'
    elif down_button.draw(screen):
        print('DOWN')
        action = 'DOWN'
    elif yaw_left_button.draw(screen):
        print('YAW-LEFT')
        action = 'YAW-LEFT'
    elif yaw_right_button.draw(screen):
        print('YAW-RIGHT')
        action = 'YAW-RIGHT'
        
    pygame.display.update()
    return keyName == action
