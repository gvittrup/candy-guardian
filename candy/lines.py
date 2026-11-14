import random

GENERIC = [
    "The candy bowl is watching... and it counted. Please put some back!",
    "Tsk tsk tsk. One for you, the rest for the ghosts. Put it back, please!",
    "Boo! The bowl's curse begins at two pieces. Return the extras and the curse is lifted.",
    "Even the skeletons only take one, and they don't even have stomachs. Put some back!",
    "Leave some for the next trick-or-treater, or the pumpkins will gossip about you.",
    "Mwa ha ha... such busy little hands! Put one back and you may pass.",
    "The spirits of this porch request... a candy refund. Thank you!",
    "Careful! That candy is enchanted. Put the extras back before it turns into broccoli!",
    "Hmmmm. I sense... too much candy. Put some back, and have a spooky night!",
    "Ooooh, the bowl is getting lonely. Give it back a piece or two, please!",
]

# {who} is filled only from COSTUMES below
TEMPLATES = [
    "Ah... {who}! Lovely costume. But even {who} must put the extra candy back!",
    "I see you, {who}. The bowl sees you too. Put some candy back, please!",
    "Hold it right there, {who}! One piece per spooky visitor. Back in the bowl, please!",
    "Ooooh, {who} with extra candy? The ghosts of this porch say: put some back!",
    "Greetings, {who}. The ancient candy rules say one piece. Return the rest, please!",
]

# CLIP label -> phrase spoken in the line
COSTUMES = {
    "witch": "a little witch",
    "wizard": "a wizard",
    "ghost": "a ghost",
    "skeleton": "a skeleton",
    "vampire": "a vampire",
    "zombie": "a zombie",
    "mummy": "a mummy",
    "superhero": "a superhero",
    "princess": "a princess",
    "fairy": "a fairy",
    "pirate": "a pirate",
    "ninja": "a ninja",
    "dinosaur": "a dinosaur",
    "pumpkin": "a pumpkin",
    "astronaut": "an astronaut",
    "robot": "a robot",
    "cat": "a kitty cat",
    "unicorn": "a unicorn",
}

GREETINGS = [
    "Happy Halloween, little ghouls! Take a few pieces... just a few. I'm watching.",
    "Welcome, trick or treaters! Help yourself to a couple of treats, but leave some for the other monsters.",
    "Happy Halloween! This bowl is enchanted. Take a few pieces, and the curse stays asleep.",
    "Ooooh, visitors! Happy Halloween! Just a few pieces each, please. The bowl remembers.",
    "Greetings, brave trick or treaters! Take a few pieces, and you may pass in peace.",
    "Happy Halloween! Grab a couple of treats, and save some for the next little monsters.",
]

STOLEN = [
    "Hey! The bowl stays on the porch! Bring it back, and nobody gets haunted.",
    "Whoa whoa whoa... the WHOLE bowl? Bring it back right now, please!",
    "The candy bowl has been kidnapped! Return it at once, little monster!",
    "Ah ah ah! That bowl is cursed if it leaves the porch. Put it back!",
]

RETURNED = [
    "Thank you! The curse is lifted. Happy Halloween!",
    "Ahh, the bowl is home. You may go in peace. Happy Halloween!",
]


class LineBook:
    def __init__(self, use_costumes):
        self.use_costumes = use_costumes
        self._last = None

    def costume_lines(self, costume):
        return [t.format(who=COSTUMES[costume]) for t in TEMPLATES]

    def texts(self):
        """Every line that can ever be spoken (for pre-rendering)."""
        out = GENERIC + STOLEN + RETURNED + GREETINGS
        if self.use_costumes:
            for c in COSTUMES:
                out += self.costume_lines(c)
        return out

    def pick(self, costume=None):
        return self.pick_from(self.costume_lines(costume) if costume in COSTUMES else GENERIC)

    def pick_from(self, pool):
        choices = [t for t in pool if t != self._last] or pool
        self._last = random.choice(choices)
        return self._last