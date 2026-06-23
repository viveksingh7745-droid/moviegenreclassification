"""
generate_data.py
Creates a realistic movie dataset with titles, overviews, and genres.
Run this first — it writes data/movies.csv
"""

import pandas as pd
import os

os.makedirs("data", exist_ok=True)

MOVIES = [
    # Action
    ("Die Hard", "A New York cop fights terrorists who take over a skyscraper on Christmas Eve.", "Action"),
    ("Mad Max: Fury Road", "In a post-apocalyptic wasteland a woman rebels against a tyrant seeking fuel and freedom.", "Action"),
    ("John Wick", "An ex-hitman comes out of retirement to track down the gangsters who took everything from him.", "Action"),
    ("The Dark Knight", "Batman faces the Joker, a criminal mastermind who plunges Gotham into chaos.", "Action"),
    ("Top Gun: Maverick", "A Navy pilot pushes the limits as a test pilot and trains young graduates for a dangerous mission.", "Action"),
    ("Mission Impossible", "An elite spy team is framed for murder and must clear their names on a globe-trotting mission.", "Action"),
    ("Speed", "A cop must keep a bomb-rigged bus above 50 mph or it will explode.", "Action"),
    ("The Bourne Identity", "A man found floating in the ocean has amnesia and discovers he is a trained CIA assassin.", "Action"),
    ("Black Hawk Down", "An elite group of soldiers are sent to Somalia to capture a warlord and face fierce combat.", "Action"),
    ("Heat", "A veteran detective and a master criminal face off in Los Angeles during a heist.", "Action"),
    ("Extraction", "A black market mercenary is hired to rescue the kidnapped son of a crime lord in Bangladesh.", "Action"),
    ("The Raid", "An elite commando squad becomes trapped in a building controlled by a ruthless crime lord.", "Action"),

    # Comedy
    ("The Grand Budapest Hotel", "A concierge teams up with a lobby boy to prove his innocence in a murder mystery.", "Comedy"),
    ("Superbad", "Two inseparable best friends try to make the most of their last days before college.", "Comedy"),
    ("Bridesmaids", "Competition between bridesmaids threatens to ruin a friendship ahead of a wedding.", "Comedy"),
    ("The Big Lebowski", "A laid-back bowler is mistaken for a millionaire and gets caught up in a kidnapping plot.", "Comedy"),
    ("Knives Out", "A master detective investigates the death of a crime novelist with a large dysfunctional family.", "Comedy"),
    ("Game Night", "A group of friends who meet for game night find themselves entangled in a real murder mystery.", "Comedy"),
    ("Step Brothers", "Two lazy middle-aged men become stepbrothers when their single parents marry each other.", "Comedy"),
    ("Groundhog Day", "A grumpy weatherman finds himself reliving the same day over and over again.", "Comedy"),
    ("The Hangover", "Three groomsmen wake up with no memory of the previous night and must find the missing groom.", "Comedy"),
    ("Anchorman", "A proud San Diego news anchor is furious when a new female reporter joins his team.", "Comedy"),
    ("Office Space", "An office worker fed up with his job hatches a scheme to defraud the company.", "Comedy"),
    ("We're the Millers", "A drug dealer creates a fake family to smuggle marijuana across the Mexican border.", "Comedy"),

    # Drama
    ("The Shawshank Redemption", "A banker is sentenced to life in prison and forms a bond with a fellow inmate over years.", "Drama"),
    ("Forrest Gump", "The life of a slow-witted man from Alabama who witnesses and influences key events in American history.", "Drama"),
    ("Schindler's List", "A German businessman saves more than a thousand Jewish refugees during the Holocaust.", "Drama"),
    ("12 Angry Men", "Twelve jurors deliberate the murder case of a young man accused of killing his father.", "Drama"),
    ("A Beautiful Mind", "The story of John Nash a brilliant mathematician who suffers from schizophrenia.", "Drama"),
    ("The Pursuit of Happyness", "A struggling salesman takes custody of his son as he searches for a better life.", "Drama"),
    ("Good Will Hunting", "A janitor at MIT has a gift for mathematics and is discovered by a professor.", "Drama"),
    ("Manchester by the Sea", "A man returns to his hometown to care for his nephew after his brother's death.", "Drama"),
    ("Whiplash", "A young ambitious drummer enrolls at a cut-throat music conservatory and faces a ruthless teacher.", "Drama"),
    ("The Imitation Game", "A mathematician helps crack the Nazi Enigma code during World War II.", "Drama"),
    ("Parasite", "Greed and class discrimination threaten a symbiotic relationship between two families.", "Drama"),
    ("Moonlight", "A young Black man growing up in Miami struggles with poverty and identity across three stages of life.", "Drama"),

    # Horror
    ("Get Out", "A Black man uncovers disturbing secrets when he meets his white girlfriend's family.", "Horror"),
    ("Hereditary", "After the death of the family matriarch strange and sinister things begin to unravel.", "Horror"),
    ("The Conjuring", "Paranormal investigators help a family terrorized by a dark presence in their farmhouse.", "Horror"),
    ("A Quiet Place", "A family struggles to survive in a post-apocalyptic world inhabited by blind monsters.", "Horror"),
    ("It", "A group of kids face an evil demonic clown that preys on children's fears.", "Horror"),
    ("Midsommar", "A couple travel to Sweden for a midsummer festival that takes a sinister turn.", "Horror"),
    ("The Babadook", "A widowed mother and her child are haunted by a monster from a mysterious storybook.", "Horror"),
    ("Sinister", "A crime writer discovers a box of disturbing home movies that put his family in danger.", "Horror"),
    ("Insidious", "A couple tries to prevent evil spirits from trapping their son's body in a comatose state.", "Horror"),
    ("Us", "A family is terrorized by doppelgangers of themselves while on vacation.", "Horror"),
    ("The Witch", "A Puritan family in New England is torn apart by superstition paranoia and witchcraft.", "Horror"),
    ("Annihilation", "A biologist signs up for a secret expedition into a mysterious quarantine zone.", "Horror"),

    # Romance
    ("Titanic", "A young woman falls in love with a penniless artist aboard the ill-fated ship.", "Romance"),
    ("La La Land", "A jazz musician and an aspiring actress fall in love while pursuing their dreams in Los Angeles.", "Romance"),
    ("Pride and Prejudice", "The story of love between Elizabeth Bennet and the wealthy Mr Darcy in 19th century England.", "Romance"),
    ("The Notebook", "A poor young man falls for a rich young woman and they are torn apart by her family.", "Romance"),
    ("Before Sunrise", "A young American man and a French woman meet on a train and spend a night together in Vienna.", "Romance"),
    ("Crazy Rich Asians", "An American-born woman is taken to Singapore to meet her boyfriend's extravagantly wealthy family.", "Romance"),
    ("About Time", "A young man discovers he can travel in time and changes the past to improve his life and love.", "Romance"),
    ("500 Days of Summer", "A hopeless romantic falls for a woman who does not believe in love and the aftermath.", "Romance"),
    ("Eternal Sunshine of the Spotless Mind", "A couple undergoes a procedure to erase each other from their memories after a bitter breakup.", "Romance"),
    ("Her", "A lonely writer develops a relationship with an AI operating system.", "Romance"),
    ("Atonement", "A young girl misidentifies a crime and tears apart the lives of her sister and a young man.", "Romance"),
    ("Call Me by Your Name", "A seventeen year old develops a relationship with a graduate student in 1980s Italy.", "Romance"),

    # Sci-Fi
    ("Inception", "A thief who enters dreams to steal secrets is tasked with planting an idea instead.", "Sci-Fi"),
    ("Interstellar", "A team of explorers travel through a wormhole in space to find a new home for humanity.", "Sci-Fi"),
    ("The Matrix", "A computer hacker discovers reality as he knows it is a simulation controlled by machines.", "Sci-Fi"),
    ("Blade Runner 2049", "A new blade runner unearths a secret that could change civilization as he knows it.", "Sci-Fi"),
    ("Arrival", "A linguist is recruited to communicate with alien beings who arrive on Earth.", "Sci-Fi"),
    ("Ex Machina", "A programmer is invited to administer a Turing test on a humanoid robot with artificial intelligence.", "Sci-Fi"),
    ("Gravity", "Two astronauts work together to survive after an accident leaves them stranded in space.", "Sci-Fi"),
    ("The Martian", "An astronaut is left behind on Mars and must survive while NASA works to rescue him.", "Sci-Fi"),
    ("Contact", "A scientist receives a signal from deep space and fights for the chance to make contact.", "Sci-Fi"),
    ("District 9", "Aliens are forced to live in a slum on Earth while a bureaucrat is infected with alien technology.", "Sci-Fi"),
    ("Moon", "An astronaut nearing the end of a solo mission on the Moon makes a frightening discovery.", "Sci-Fi"),
    ("Dune", "A noble family becomes embroiled in a war for control of a desert planet's precious resource.", "Sci-Fi"),

    # Thriller
    ("Gone Girl", "A man becomes the prime suspect when his wife mysteriously disappears on their anniversary.", "Thriller"),
    ("Se7en", "Two detectives hunt a serial killer who uses the seven deadly sins as his motives.", "Thriller"),
    ("Prisoners", "A father takes the law into his own hands when his daughter and her friend go missing.", "Thriller"),
    ("Zodiac", "Detectives and reporters are obsessed with catching the Zodiac Killer in 1960s San Francisco.", "Thriller"),
    ("No Country for Old Men", "A hunter stumbles upon drug money and is pursued by a relentless killer in rural Texas.", "Thriller"),
    ("Nightcrawler", "A driven young man learns the business of crime journalism in Los Angeles.", "Thriller"),
    ("Sicario", "An FBI agent is recruited to join a task force targeting a Mexican drug cartel.", "Thriller"),
    ("Oldboy", "A man is imprisoned for fifteen years without explanation and seeks revenge on his captor.", "Thriller"),
    ("The Girl with the Dragon Tattoo", "A journalist and a hacker investigate the disappearance of a wealthy patriarch's niece.", "Thriller"),
    ("Shutter Island", "A U.S. marshal investigates the disappearance of a patient from a hospital for the criminally insane.", "Thriller"),
    ("Memento", "A man with short-term memory loss uses notes and tattoos to track down his wife's killer.", "Thriller"),
    ("Parasite", "Two families become entangled in a deadly game of deception and class warfare.", "Thriller"),

    # Animation
    ("Spirited Away", "A young girl enters a spirit world and must work to free herself and her transformed parents.", "Animation"),
    ("The Lion King", "A young lion prince flees his kingdom only to learn the true meaning of responsibility.", "Animation"),
    ("Finding Nemo", "A clownfish ventures across the ocean to find his missing son who was taken by a scuba diver.", "Animation"),
    ("Up", "A widowed man ties balloons to his house to fly to South America with a young stowaway.", "Animation"),
    ("WALL-E", "A small robot left on an abandoned Earth falls in love with a probe sent from a spaceship.", "Animation"),
    ("Coco", "A young musician is transported to the land of the dead and seeks his music idol great-great-grandfather.", "Animation"),
    ("Princess Mononoke", "A young prince becomes involved in a war between forest gods and humans who consume nature.", "Animation"),
    ("Toy Story", "A cowboy doll is threatened when a new spaceman toy becomes the owner's favorite.", "Animation"),
    ("Inside Out", "A young girl's emotions personified as characters guide her through a difficult move to a new city.", "Animation"),
    ("Your Name", "Two strangers find they are linked in a bizarre way experiencing each other's lives through dreams.", "Animation"),
    ("Howl's Moving Castle", "A young woman is cursed by a witch and seeks the help of a wizard in his moving castle.", "Animation"),
    ("Kubo and the Two Strings", "A young boy with magical powers embarks on a quest to find a legendary suit of armor.", "Animation"),
]

def main():
    df = pd.DataFrame(MOVIES, columns=["title", "overview", "genre"])
    df.index.name = "id"
    df.to_csv("data/movies.csv")
    print(f"Dataset created: {len(df)} movies across {df['genre'].nunique()} genres")
    print("\nGenre distribution:")
    print(df["genre"].value_counts().to_string())
    print(f"\nSaved → data/movies.csv")

if __name__ == "__main__":
    main()
