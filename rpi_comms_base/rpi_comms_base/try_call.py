import lgpio

ena1 = 19
enb1 = 26

chip = lgpio.gpiochip_open(0)

lgpio.gpio_claim_input(chip, ena1, 1)
lgpio.gpio_claim_input(chip, enb1, 1)

previous_a = 0
en_count = 0

def encoder_callback(gpiochip, gpio, level, tick):
    print("en callback triggered")
    global previous_a, en_count
    current_a = lgpio.gpio_read(chip, ena1)
    current_b = lgpio.gpio_read(chip, enb1)
    print("current: ", current_a)

    if current_a != previous_a:
        if current_a == current_b:
            en_count += 1
        else:
            en_count -= 1
        previous_a = current_a
        print(f"Encoder Count: {en_count}")

def call_encoder_int():
    call1 = lgpio.callback(chip, ena1, lgpio.BOTH_EDGES, encoder_callback)
    print("callback 1")
    call2 = lgpio.callback(chip, enb1, lgpio.BOTH_EDGES, encoder_callback)
    print("callback 2")
    return call1, call2

def main():
    call1, call2 = call_encoder_int()
    while True:
        x = input()
        print(en_count)

        if x == 'e':
            lgpio.gpiochip_close(chip)
            print("LGPIO Clean up")
            break

        print(call1, call2)

if __name__ == "__main__":
    main()