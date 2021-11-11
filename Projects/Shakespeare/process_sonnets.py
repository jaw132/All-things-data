def main():

    roman_list = roman_numerals()

    all_sonnets = open("sonnets/all_sonnets.txt", "r")
    full_text = ""
    for line in all_sonnets:
        full_text = full_text + line

    processed_sonnets = full_text.replace("\n\n", " ")
    all_sonnets.close()

    '''create txt file with all sonnets in it and also 
    want to save each individual sonnet'''

    # chop up sonnets
    roman_ind=0
    look_up = roman_list[roman_ind]+"."
    ind_sonnet, processed_sonnets_v2 = "", ""

    for line in processed_sonnets.splitlines():
        if line == look_up:
            if roman_ind > 0:
                file = open("sonnet" + str(roman_ind) + ".txt", "w")
                file.writelines(ind_sonnet)
                file.close()
            ind_sonnet = ""
            roman_ind+=1
            if roman_ind < 154:
                look_up = roman_list[roman_ind]+"."
        else:
            processed_sonnets_v2 = processed_sonnets_v2+"\n"+line
            ind_sonnet = ind_sonnet+"\n"+line

    final_sonnet = open("sonnet" + str(roman_ind) + ".txt", "w")
    final_sonnet.writelines(ind_sonnet)
    final_sonnet.close()

    final_file = open("sonnets/processed_sonnets.txt", "w")
    final_file.writelines(processed_sonnets_v2)
    final_file.close()


def roman_numerals():

    last_digit_map = {1: 'I',
                      2: 'II',
                      3: 'III',
                      4: 'IV',
                      5: 'V',
                      6: 'VI',
                      7: 'VII',
                      8: 'VIII',
                      9: 'IX',
                      0: ''}
    second_digit_map = {1: 'X',
                        2: 'XX',
                        3: 'XXX',
                        4: 'XL',
                        5: 'L',
                        6: 'LX',
                        7: 'LXX',
                        8: 'LXXX',
                        9: 'XC',
                        0: ''}
    roman_list = []
    for number in range(1, MAX_SONNET+1):
        if number < 10:
            roman_list.append(last_digit_map[number])
        elif number < 100:
            last_digit = number % 10
            second_digit = number//10
            roman_list.append(second_digit_map[second_digit]+last_digit_map[last_digit])
        else:
            last_digit = number % 10
            second_digit = (number // 10) % 10
            roman_list.append('C'+second_digit_map[second_digit] + last_digit_map[last_digit])
    return roman_list


if __name__ == '__main__':
    MAX_SONNET = 154
    main()
