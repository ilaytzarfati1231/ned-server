from pyformlang.regular_expression import PythonRegex
import Automata
import networkx as nx
save_object=[]


def option1():
    regex = PythonRegex(input("Enter the regex: "))
    nfa = regex.to_epsilon_nfa().minimize()
    save_object.append(nfa)
    print("Automata added successfully!")


def option2():
    print("which automata do you want to print?")
    for i in range(len(save_object)):
        print(i, end=" ")
    print()
    choice = int(input())
    saved = save_object[choice]
    if not isinstance(saved, nx.DiGraph):
        Automata.visualize_automata(saved, "automata" + str(choice) )
    else:
        Automata.visualize_automata_graph(saved, "automata" + str(choice))


def option3():
    print("which automata do you want to product?")
    for i in range(len(save_object)):
        print(i, end=" ")
    print()
    choice1 = int(input())
    print("which automata do you want to product?")
    for i in range(len(save_object)):
        print(i, end=" ")
    print()
    choice2 = int(input())
    automata = Automata.automata_edit_distance_graph(save_object[choice1], save_object[choice2])
    save_object.append(automata)
    print("Automata producted successfully!")


def option4():
    print("which automata do you want to inf?")
    for i in range(len(save_object)):
        print(i, end=" ")
    print()
    choice = int(input())
    automata = save_object[choice]
    print("The automata is inf: ", Automata.inf_inf(automata))


def main():
    regex = PythonRegex("(aa)*")
    nfa = regex.to_epsilon_nfa().minimize()
    save_object.append(nfa)
    regex2 = PythonRegex("(ab)*")
    nfa2 = regex2.to_epsilon_nfa().minimize()
    save_object.append(nfa2)
    product = Automata.automata_edit_distance_graph(save_object[0], save_object[1])
    save_object.append(product)
    print("Welcome! Please select an option:")
    print("1. add new automata by regex")
    print("2. print automata")
    print("3. product automatas")
    print("4. inf inf works only on product !")
    print("5. exit")

    choice = input("Enter the number of your choice: ")
    while choice != "5":
        if choice == "1":
            option1()
        elif choice == "2":
            print("You chose Option 2")
            option2()
            # Add code for Option 2 here
        elif choice == "3":
            print("You chose Option 3")
            option3()
            # Add code for Option 3 here

        elif choice == "4":
            print("You chose Option 4")
            option4()
            # Add code to exit the program
        elif choice == "5":
            print("Exiting program...")

        else:
            print("Invalid choice. Please select a valid option.")
        choice = input("Enter the number of your choice: ")


if __name__ == "__main__":
    main()


