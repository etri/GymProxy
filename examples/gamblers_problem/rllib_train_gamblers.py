from typing import Optional

NUM_STEPS = 100
PROB_HEAD = 0.5
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

config = {'num_steps': NUM_STEPS,
          'prob_head': PROB_HEAD,
          'initial_capital': INITIAL_CAPITAL,
          'winning_capital': WINNING_CAPITAL}

def test2(**kwargs):
    print(kwargs)

def test(kwargs: Optional[dict] = None):
    print(kwargs)
    test2(**kwargs)

def main():
    test(config)

if __name__ == "__main__":
    main()
