class Organism:
    dna = {}

    def breed_with(self,mate: Organism) -> bool:
        pass

class Human(Organism):
    dna = {
        "intelligence": 0,
        "strength": 0,
        "wisdom": 0
    }

    


humans = [Human()]
humans[0].dna = {}
