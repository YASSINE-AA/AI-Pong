import pygame
from random import randint
BLACK = (0, 0, 0)

class Ball(pygame.sprite.Sprite):
    def __init__(self, color, width, height):
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)

        pygame.draw.rect(self.image, color, [0, 0, width, height])
        self.velocity = [2, 2]

        self.rect = self.image.get_rect()
    def bounce(self): 
        self.velocity[0] = -self.velocity[0]

    def update(self):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]