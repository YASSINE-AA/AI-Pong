from re import L
from turtle import width
import pygame
from components.Paddle import Paddle
from components.Ball import Ball
from settings import *
import neat
import os
import sys
import pickle

class PongGame:

    def __init__(self, window, width, height):
        pygame.init()


        # clock is used to control how fast the screen updates a second.
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.gameFont = pygame.font.SysFont("Comic Sans MS", 30)


        self.playerA_score = 0
        self.playerB_score = 0

        self.paddleA = Paddle(WHITE, 10, 100)
        self.paddleA.rect.x = 20
        self.paddleA.rect.y = 200

        self.paddleB = Paddle(WHITE, 10, 100)
        self.paddleB.rect.x = 670
        self.paddleB.rect.y = 200
        self.left_hits = 0
        self.right_hits = 0
        self.ball = Ball(WHITE, 10, 10)
        self.ball.rect.x = 350
        self.ball.rect.y = 250
        self.Running = True
        self.all_sprites_list = pygame.sprite.Group()
        self.all_sprites_list.add(self.paddleA)
        self.all_sprites_list.add(self.paddleB)
        self.all_sprites_list.add(self.ball)
 
        self.screen = window  # Initialize Screen
        pygame.display.set_caption("Pong")
    
      

    def draw(self):
        playerA_scoreText = self.gameFont.render(
            f"{self.left_hits}", False, WHITE)
        playerB_scoreText = self.gameFont.render(
            f"{self.right_hits}", False, WHITE)
        self.ball.update()
        self.screen.fill(BLACK)
        self.screen.blit(playerA_scoreText, (20, 20))
        self.screen.blit(playerB_scoreText, (659, 19))
        pygame.draw.line(self.screen, WHITE, [349, 0], [349, 500], 2)
        self.all_sprites_list.draw(self.screen)
    def loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.Running = False
            elif event.type == pygame.K_x:
                self.Running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.paddleA.moveUp(10)
        if keys[pygame.K_s]:
            self.paddleA.moveDown(10)
        if keys[pygame.K_UP]:
            self.paddleB.moveUp(10)
        if keys[pygame.K_DOWN]:
            self.paddleB.moveDown(10)

        self.all_sprites_list.update()
        if pygame.sprite.collide_mask(self.ball, self.paddleA) or pygame.sprite.collide_mask(self.ball, self.paddleB):
            self.ball.bounce()



        if pygame.sprite.collide_mask(self.ball, self.paddleA):
            self.left_hits +=1
        if pygame.sprite.collide_mask(self.ball, self.paddleB):
            self.right_hits += 1

     

        if self.ball.rect.x >= 690:
            self.ball.rect.x = 350
            self.ball.rect.y = 250
            # Increment Score
            self.playerA_score += 1

        if self.ball.rect.x <= 0:
            self.ball.rect.x = 350
            self.ball.rect.y = 250
            # Increment Score
            self.playerB_score += 1

        if self.ball.rect.y > 490:
            self.ball.velocity[1] = -self.ball.velocity[1]
        if self.ball.rect.y < 0:
            self.ball.velocity[1] = -self.ball.velocity[1]
        self.draw()
        # Update the Screen.
        pygame.display.flip()

        # Limit to 60 frames per second.
        self.clock.tick(75)
        
   
    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        run = True
        pygame.init()
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            output1 = net1.activate((self.paddleA.rect.y, self.ball.rect.y, abs(
                self.paddleA.rect.x - self.ball.rect.x)))
            decision1 = output1.index(max(output1)) 

            output2 = net2.activate((self.paddleB.rect.y, self.ball.rect.y, abs(
                self.paddleB.rect.x - self.ball.rect.x)))
            decision2 = output2.index(max(output2))

            if(decision1 == 0):
                pass

            elif decision1 == 1:
                self.paddleA.moveUp(6)
            else:
                self.paddleA.moveDown(6)

            if(decision2 == 0):
                 pass

            elif decision2 == 1:
                self.paddleB.moveUp(6)
            else:
                self.paddleB.moveDown(6)
                
            print(output1, output2, sep="|")

            game_info = self.loop()
            pygame.display.update()
     
            if self.playerA_score >= 1 or self.playerB_score >= 1 or self.left_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                
        
    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += self.left_hits
        genome2.fitness += self.right_hits



def eval_genomes(genomes, config):
    # Set Fitness for each genome
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    for i, (genome_id, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = PongGame(window, 700, 500)
            game.train_ai(genome1, genome2, config)


def run_neat(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    winner = p.run(eval_genomes, 1)
    #Dump Genome into pickle file
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

def test_ai(config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))
    with open("best.pickle", "wb") as f:
        winner = pickle.load(f)
    game = PongGame(window, width, height)
    game.test_ai(winner, config)

if __name__ == "__main__":

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    run_neat(config)

