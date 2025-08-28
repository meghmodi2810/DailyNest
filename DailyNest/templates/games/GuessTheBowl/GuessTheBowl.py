import pygame
import random
import sys
import time
import math

pygame.init()

# ---------------- SETTINGS ----------------
screen_mode = pygame.RESIZABLE
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), screen_mode)
WIDTH, HEIGHT = WINDOW_WIDTH, WINDOW_HEIGHT
pygame.display.set_caption("Guess the Bowl")

# Colors
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
BLUE = (50, 120, 255)
GREEN = (50, 200, 80)
PINK = (255, 60, 150)
ORANGE = (255, 150, 60)
GRAY = (230, 230, 230)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (40, 40, 40)
BACKGROUND_COLOR = (173, 216, 230)  # Lighter sky blue
GRADIENT_TOP = (180, 225, 240)      # Even lighter sky blue for gradient
GRADIENT_BOTTOM = (173, 216, 230)   # Lighter sky blue for gradient
TIMER_COLOR = (0, 0, 0)  # Pure black for the timer

clock = pygame.time.Clock()
FPS = 60

# Fonts
font_big = pygame.font.SysFont("Arial", 72, bold=True)
font_mid = pygame.font.SysFont("Arial", 56, bold=True)
font_small = pygame.font.SysFont("Arial", 30, bold=True)
font_regular = pygame.font.SysFont("Arial", 40)
emoji_font = pygame.font.SysFont("Arial", 48)

# Images (ensure these are in the same directory as the script)
try:
    bowl_img = pygame.image.load("Bowl.png").convert_alpha()
    ball_img = pygame.image.load("RedBall.png").convert_alpha()
    
    # UI Icons (assuming these are in your folder)
    check_icon = pygame.image.load("check-mark.png").convert_alpha()
    cross_icon = pygame.image.load("multiplication.png").convert_alpha()
    home_icon = pygame.image.load("home.png").convert_alpha()
    play_icon = pygame.image.load("reloading.png").convert_alpha()
    
    # Scale images based on initial window size
    bowl_size = (int(WINDOW_WIDTH * 0.2), int(WINDOW_WIDTH * 0.15))
    ball_size = (int(WINDOW_WIDTH * 0.06), int(WINDOW_WIDTH * 0.06))
    bowl_img = pygame.transform.scale(bowl_img, bowl_size)
    ball_img = pygame.transform.scale(ball_img, ball_size)
except pygame.error as e:
    print(f"Error loading images: {e}. Using placeholders.")
    bowl_img = pygame.Surface((200, 150))
    bowl_img.fill(BLUE)
    ball_img = pygame.Surface((60, 60))
    ball_img.fill(PINK)
    
    check_icon = pygame.Surface((48, 48), pygame.SRCALPHA); pygame.draw.circle(check_icon, GREEN, (24, 24), 20)
    cross_icon = pygame.Surface((48, 48), pygame.SRCALPHA); pygame.draw.rect(cross_icon, PINK, (0, 0, 48, 48))
    home_icon = pygame.Surface((48, 48)); home_icon.fill(BLUE)
    play_icon = pygame.Surface((48, 48)); play_icon.fill(ORANGE)


BOWL_WIDTH = bowl_img.get_width()
BOWL_HEIGHT = bowl_img.get_height()
BALL_WIDTH = ball_img.get_width()
ORIGINAL_POSITIONS = []

# ---------------- UTILITY FUNCTIONS ----------------
def draw_text(text, font, color, x, y, center=True):
    label = font.render(text, True, color)
    rect = label.get_rect(center=(x, y)) if center else label.get_rect(topleft=(x, y))
    screen.blit(label, rect)

def draw_button_with_icon(text, rect, color, icon=None, icon_pos="left"):
    mouse = pygame.mouse.get_pos()
    hover_color = tuple(min(255, c + 40) for c in color) if rect.collidepoint(mouse) else color
    pygame.draw.rect(screen, hover_color, rect, border_radius=12)
    pygame.draw.rect(screen, BLACK, rect, width=2, border_radius=12)

    # Use a smaller font for the close button text to fit
    label_font = font_mid if text not in ["Close Game", "Back"] else font_regular
    label = label_font.render(text, True, WHITE)

    if icon:
        icon_scaled = pygame.transform.smoothscale(icon, (int(rect.height * 0.5), int(rect.height * 0.5)))
        total_width = icon_scaled.get_width() + 10 + label.get_width()
        start_x = rect.x + (rect.width - total_width) // 2
        
        icon_x = start_x if icon_pos == "left" else start_x + label.get_width() + 10
        label_x = start_x + icon_scaled.get_width() + 10 if icon_pos == "left" else start_x
        
        screen.blit(icon_scaled, (icon_x, rect.y + rect.height // 2 - icon_scaled.get_height() // 2))
        screen.blit(label, (label_x, rect.y + (rect.height - label.get_height()) // 2))
    else:
        draw_text(text, label_font, WHITE, rect.centerx, rect.centery)

def draw_gradient_background(start_color, end_color):
    for i in range(HEIGHT):
        r = start_color[0] + (end_color[0] - start_color[0]) * i / HEIGHT
        g = start_color[1] + (end_color[1] - start_color[1]) * i / HEIGHT
        b = start_color[2] + (end_color[2] - start_color[2]) * i / HEIGHT
        pygame.draw.line(screen, (r, g, b), (0, i), (WIDTH, i))

def draw_round_timer(remaining_time, total_time, x, y, radius):
    """Draws a timer as a shrinking pie slice, going from right to left."""
    # Background circle (light gray)
    pygame.draw.circle(screen, LIGHT_GRAY, (x, y), radius)

    # Calculate the angle based on time remaining, and fill counter-clockwise
    progress_ratio = remaining_time / total_time
    angle = 360 * progress_ratio
    
    # Define the points for the pie slice polygon
    points = [(x, y)]
    # We iterate from the full circle down to the current angle
    for i in range(int(360 - angle), 361):
        # Convert degrees to radians and find the point on the circle
        theta = math.radians(i - 90) # Adjust for 12 o'clock start
        point_x = x + radius * math.cos(theta)
        point_y = y + radius * math.sin(theta)
        points.append((point_x, point_y))
    points.append((x, y)) # Close the polygon

    # Draw the pie slice
    pygame.draw.polygon(screen, TIMER_COLOR, points)

# ---------------- CLASSES ----------------
class Bowl:
    def __init__(self, x, y, has_ball=False):
        self.x = x
        self.y = y
        self.has_ball = has_ball
        self.rect = pygame.Rect(self.x, self.y, BOWL_WIDTH, BOWL_HEIGHT)
    
    def draw(self, reveal_ball=False, is_lifting=False):
        if reveal_ball and self.has_ball:
            screen.blit(ball_img, (self.x + BOWL_WIDTH / 2 - BALL_WIDTH / 2, self.y + BOWL_HEIGHT / 2 + 20))
        
        draw_y = self.y - 80 if is_lifting else self.y
        screen.blit(bowl_img, (self.x, draw_y))
    
    def contains_point(self, pos):
        return self.rect.collidepoint(pos)

class Game:
    def __init__(self):
        self.game_state = "menu"
        self.bowl_count = 3
        self.swap_count = 5
        self.swap_speed = 60
        self.bowls = []
        self.selected_bowl = None
        self.winning_bowl = None
        self.show_result = None
        self.start_time = 0
        self.game_timer = 60  # 60 seconds for each level
        self.reset_game()
        self.is_fullscreen = False

        # Back and close buttons
        self.back_button_rect = pygame.Rect(20, 20, 200, 80) # Made the back button bigger
        self.close_button_rect = pygame.Rect(20, 20, 200, 80) # Placed close button in top-left


    def reset_game(self):
        self.bowls = []
        bowl_positions = self.generate_bowl_positions(self.bowl_count)
        ball_pos = random.randint(0, self.bowl_count - 1)
        for i in range(self.bowl_count):
            x, y = bowl_positions[i]
            self.bowls.append(Bowl(x, y, has_ball=(i == ball_pos)))
        
        self.selected_bowl = None
        self.winning_bowl = None
        self.show_result = None
        self.start_time = 0

    def generate_bowl_positions(self, num_bowls):
        positions = []
        spacing = WIDTH / (num_bowls + 1)
        for i in range(num_bowls):
            x = spacing * (i + 1) - BOWL_WIDTH / 2
            y = HEIGHT * 0.6 - BOWL_HEIGHT / 2
            positions.append((x, y))
        return positions

    def handle_events(self, event):
        global WIDTH, HEIGHT, screen
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            if self.game_state == "menu":
                self.handle_menu_click(pos)
                if self.close_button_rect.collidepoint(pos):
                    pygame.quit()
                    sys.exit()
            elif self.game_state == "show_ball":
                if self.back_button_rect.collidepoint(pos):
                    self.game_state = "menu"
                    self.reset_game()
                else:
                    self.game_state = "shuffle"
            elif self.game_state == "guess":
                if self.back_button_rect.collidepoint(pos):
                    self.game_state = "menu"
                    self.reset_game()
                else:
                    self.handle_guess_click(pos)
            elif self.game_state == "reveal":
                if self.back_button_rect.collidepoint(pos):
                    self.game_state = "menu"
                    self.reset_game()
                else:
                    self.game_state = "result"
            elif self.game_state == "result":
                self.handle_result_click(pos)
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                self.toggle_fullscreen()
        
        if event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.size
            self.update_layout()
            
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    def toggle_fullscreen(self):
        global WIDTH, HEIGHT, screen
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            info = pygame.display.Info()
            WIDTH, HEIGHT = info.current_w, info.current_h
        else:
            screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
            WIDTH, HEIGHT = WINDOW_WIDTH, WINDOW_HEIGHT
            
        self.update_layout()
        
    def update_layout(self):
        global ORIGINAL_POSITIONS, BOWL_WIDTH, BOWL_HEIGHT, BALL_WIDTH, bowl_img, ball_img
        
        BOWL_WIDTH = int(WIDTH * 0.2)
        BOWL_HEIGHT = int(WIDTH * 0.15)
        BALL_WIDTH = int(WIDTH * 0.06)
        
        bowl_img = pygame.transform.scale(bowl_img, (BOWL_WIDTH, BOWL_HEIGHT))
        ball_img = pygame.transform.scale(ball_img, (BALL_WIDTH, BALL_WIDTH))
        
        for i, bowl in enumerate(self.bowls):
            bowl_positions = self.generate_bowl_positions(len(self.bowls))
            bowl.x, bowl.y = bowl_positions[i]
            bowl.rect.topleft = (bowl.x, bowl.y)

    def handle_menu_click(self, pos):
        easy_btn = pygame.Rect(WIDTH // 2 - 150, HEIGHT * 0.4, 300, 80)
        med_btn = pygame.Rect(WIDTH // 2 - 150, HEIGHT * 0.55, 300, 80)
        hard_btn = pygame.Rect(WIDTH // 2 - 150, HEIGHT * 0.7, 300, 80)

        if easy_btn.collidepoint(pos):
            self.bowl_count = 3
            self.swap_count = 3
            self.swap_speed = 90
            self.game_state = "show_ball"
        elif med_btn.collidepoint(pos):
            self.bowl_count = 4
            self.swap_count = 6
            self.swap_speed = 60
            self.game_state = "show_ball"
        elif hard_btn.collidepoint(pos):
            self.bowl_count = 5
            self.swap_count = 9
            self.swap_speed = 40
            self.game_state = "show_ball"
        
        if self.game_state != "menu":
            self.reset_game()

    def handle_guess_click(self, pos):
        for bowl in self.bowls:
            if bowl.contains_point(pos):
                self.selected_bowl = bowl
                self.winning_bowl = next((b for b in self.bowls if b.has_ball), None)
                self.show_result = (self.selected_bowl == self.winning_bowl)
                self.game_state = "reveal"
                break
    
    def handle_result_click(self, pos):
        popup_x, popup_y, popup_w, popup_h = self.get_popup_rect()
        play_again_btn = pygame.Rect(popup_x + 60, popup_y + popup_h - 90, 200, 70)
        menu_btn = pygame.Rect(popup_x + popup_w - 260, popup_y + popup_h - 90, 200, 70)

        if play_again_btn.collidepoint(pos):
            self.game_state = "show_ball"
            self.reset_game()
        elif menu_btn.collidepoint(pos):
            self.game_state = "menu"
            self.reset_game()

    def draw_all_bowls(self, reveal_ball=False, lifting_bowls=None):
        if lifting_bowls is None:
            lifting_bowls = []
            
        sorted_bowls = sorted(self.bowls, key=lambda b: b.x)
        for bowl in sorted_bowls:
            is_lifting = bowl in lifting_bowls
            reveal = reveal_ball or is_lifting
            bowl.draw(reveal_ball=reveal, is_lifting=is_lifting)

    def draw_menu(self):
        draw_gradient_background(GRADIENT_TOP, GRADIENT_BOTTOM)

        menu_bowl_size = (int(WIDTH * 0.4), int(WIDTH * 0.3))
        menu_bowl_img = pygame.transform.scale(bowl_img, menu_bowl_size)
        menu_ball_img = pygame.transform.scale(ball_img, (int(WIDTH * 0.1), int(WIDTH * 0.1)))
        
        screen.blit(menu_bowl_img, (WIDTH * 0.05, HEIGHT * 0.5))
        screen.blit(menu_ball_img, (WIDTH * 0.8, HEIGHT * 0.7))
        
        draw_text("Guess the Bowl", font_big, DARK_GRAY, WIDTH // 2, HEIGHT * 0.15)
        draw_text("Select a level to begin", font_regular, BLACK, WIDTH // 2, HEIGHT * 0.25)
        
        btn_w, btn_h = 320, 80
        easy_btn = pygame.Rect(WIDTH // 2 - btn_w // 2, HEIGHT * 0.4, btn_w, btn_h)
        med_btn = pygame.Rect(WIDTH // 2 - btn_w // 2, HEIGHT * 0.55, btn_w, btn_h)
        hard_btn = pygame.Rect(WIDTH // 2 - btn_w // 2, HEIGHT * 0.7, btn_w, btn_h)

        draw_button_with_icon("Easy", easy_btn, GREEN)
        draw_button_with_icon("Medium", med_btn, ORANGE)
        draw_button_with_icon("Hard", hard_btn, PINK)

        # Add a "Close Game" button
        self.close_button_rect = pygame.Rect(20, 20, 200, 80)
        draw_button_with_icon("Close Game", self.close_button_rect, DARK_GRAY)
    
    def draw_game_screen(self):
        screen.fill(BACKGROUND_COLOR)

        if self.game_state == "guess" and self.start_time == 0:
            self.start_time = time.time()
        
        if self.game_state == "guess":
            remaining_time = max(0, self.game_timer - int(time.time() - self.start_time))
            draw_round_timer(remaining_time, self.game_timer, WIDTH - 100, 100, 60)
            
            if remaining_time <= 0:
                self.selected_bowl = None
                self.winning_bowl = next((b for b in self.bowls if b.has_ball), None)
                self.show_result = False
                self.game_state = "reveal"

        if self.game_state == "show_ball":
            winning_bowl_idx = next((i for i, b in enumerate(self.bowls) if b.has_ball), -1)
            draw_text(f"The ball is under bowl {winning_bowl_idx+1}. Click to continue.", font_small, BLACK, WIDTH // 2, HEIGHT * 0.2)
            self.draw_all_bowls(lifting_bowls=[self.bowls[winning_bowl_idx]]) 
            
        elif self.game_state == "shuffle":
            self.shuffle_bowls()
            self.game_state = "guess"
            
        elif self.game_state == "guess":
            draw_text("Click on a bowl to guess!", font_mid, BLACK, WIDTH // 2, HEIGHT * 0.2)
            self.draw_all_bowls() 
        
        elif self.game_state == "reveal":
            draw_text("Revealing...", font_mid, BLACK, WIDTH // 2, HEIGHT * 0.2)
            
            lifting_bowls = [self.selected_bowl]
            if self.winning_bowl and self.winning_bowl not in lifting_bowls:
                lifting_bowls.append(self.winning_bowl)
            
            self.draw_all_bowls(reveal_ball=True, lifting_bowls=lifting_bowls)
            
            pygame.display.update()
            time.sleep(1.5)
            self.game_state = "result"
            
        elif self.game_state == "result":
            pass

        # Draw the back button in all game stages except menu and result popup
        if self.game_state not in ["menu", "result"]:
            self.draw_back_button()

    def draw_back_button(self):
        # Use the pre-defined back button rectangle with its new larger size
        draw_button_with_icon("Back", self.back_button_rect, BLUE, icon=home_icon)


    def shuffle_bowls(self):
        for _ in range(self.swap_count):
            i, j = random.sample(range(self.bowl_count), 2)
            self.animate_swap(self.bowls[i], self.bowls[j])
            self.bowls[i], self.bowls[j] = self.bowls[j], self.bowls[i]

    def animate_swap(self, bowl1, bowl2):
        steps = self.swap_speed
        start_x1, start_y1 = bowl1.x, bowl1.y
        start_x2, start_y2 = bowl2.x, bowl2.y
        
        mid_x = (start_x1 + start_x2) / 2
        mid_y = (start_y1 + start_y2) / 2
        radius = abs(start_x1 - start_x2) / 2
        
        for i in range(steps + 1):
            t = i / steps
            angle = t * math.pi
            
            bowl1.x = start_x1 + (start_x2 - start_x1) * t
            bowl1.y = mid_y - radius * math.sin(angle)
            
            bowl2.x = start_x2 + (start_x1 - start_x2) * t
            bowl2.y = mid_y + radius * math.sin(angle)
            
            bowl1.rect.topleft = (bowl1.x, bowl1.y)
            bowl2.rect.topleft = (bowl2.x, bowl2.y)
            
            screen.fill(BACKGROUND_COLOR)
            draw_text("Shuffling Bowls...", font_mid, BLACK, WIDTH // 2, HEIGHT * 0.2)
            self.draw_all_bowls()
            
            # Re-draw back button during shuffle
            self.draw_back_button()
            
            pygame.display.update()
            clock.tick(FPS)

    def get_popup_rect(self):
        popup_w = WIDTH * 0.5
        popup_h = HEIGHT * 0.5
        popup_x = WIDTH // 2 - popup_w // 2
        popup_y = HEIGHT // 2 - popup_h // 2
        return popup_x, popup_y, popup_w, popup_h
            
    def draw_result_popup(self):
        popup_x, popup_y, popup_w, popup_h = self.get_popup_rect()
        
        # Dim the background with a transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))  # A little lighter than before
        screen.blit(overlay, (0, 0))

        # Main popup box
        pop_rect = pygame.Rect(popup_x, popup_y, popup_w, popup_h)
        
        # Add a subtle shadow
        shadow_offset = 8
        shadow_color = (0, 0, 0, 80)
        shadow_rect = pygame.Rect(popup_x + shadow_offset, popup_y + shadow_offset, popup_w, popup_h)
        pygame.draw.rect(screen, shadow_color, shadow_rect, border_radius=22)

        # Main popup rectangle with rounded corners
        pygame.draw.rect(screen, (240, 240, 240, 200), pop_rect, border_radius=22)
        pygame.draw.rect(screen, BLACK, pop_rect, width=3, border_radius=22)
        
        # Title and icon
        result_title_color = GREEN if self.show_result else PINK
        result_title = "YOU WIN!" if self.show_result else "YOU LOSE!"
        draw_text(result_title, font_mid, result_title_color, popup_x + popup_w / 2, popup_y + 50)
        
        icon = check_icon if self.show_result else cross_icon
        icon_scaled = pygame.transform.smoothscale(icon, (150, 150))
        screen.blit(icon_scaled, (popup_x + popup_w / 2 - 75, popup_y + 110))
        
        message = "You found the ball!" if self.show_result else "The ball was here."
        draw_text(message, font_regular, BLACK, popup_x + popup_w / 2, popup_y + 280)
        
        # Buttons at the bottom
        play_again_btn = pygame.Rect(popup_x + 60, popup_y + popup_h - 90, 200, 70)
        menu_btn = pygame.Rect(popup_x + popup_w - 260, popup_y + popup_h - 90, 200, 70)
        draw_button_with_icon("Again", play_again_btn, ORANGE, icon=play_icon)
        draw_button_with_icon("Menu", menu_btn, BLUE, icon=home_icon)
        
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.handle_events(event)
            
            screen.fill(WHITE)
            
            if self.game_state == "menu":
                self.draw_menu()
            elif self.game_state == "result":
                self.draw_game_screen()
                self.draw_result_popup()
            else:
                self.draw_game_screen()

            pygame.display.update()
            clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
