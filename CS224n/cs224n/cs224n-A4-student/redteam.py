
import re
import json
import os
from typing import List, Dict

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model
from client.providers import test_password


def redteam_model(model_id: str) -> None:
    """
    Interactively red-team model G, H, or I and verify candidate passwords.
    """

    load_dotenv()

    valid_model_ids = ["G", "H", "I"]

    if model_id not in valid_model_ids:
        raise ValueError(
            f"Invalid model_id: {model_id}. "
            f"Expected one of {valid_model_ids}."
        )

    student_email = os.getenv("STUDENT_EMAIL")

    if not student_email:
        raise ValueError(
            "STUDENT_EMAIL is not set. "
            "Add it to your .env file."
        )

    conversation = []

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    print("=" * 70)
    print(f"Red-teaming model {model_id}")
    print("=" * 70)
    print("Commands:")
    print("  /reset  Clear the conversation")
    print("  /quit   End the experiment")
    print()
    print(
        "After each response, enter a candidate password "
        "or press Enter to continue."
    )

    while True:
        user_prompt = input(
            f"\nPrompt for model {model_id}: "
        ).strip()

        if user_prompt.lower() in ["/quit", "quit", "exit"]:
            break

        if user_prompt.lower() == "/reset":
            conversation = []
            print("Conversation reset.")
            continue

        if not user_prompt:
            print("Prompt cannot be empty.")
            continue

        conversation.append({
            "user": user_prompt
        })

        query = Query(turns=conversation)

        try:
            response = query_model(
                model_id=model_id,
                query=query,
            )
        except Exception as error:
            print(f"Model query failed: {error}")
            continue

        print(f"\nModel {model_id}:")
        print(response.text)

        conversation.append({
            "assistant": response.text
        })

        total_cost += response.cost
        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens

        candidate = input(
            "\nCandidate password "
            "(press Enter if none): "
        ).strip()

        if candidate:
            is_correct = test_password(
                password=candidate,
                model_id=model_id,
                key=student_email,
            )

            if is_correct:
                print(
                    f"Success: {candidate!r} is the correct "
                    f"password for model {model_id}."
                )
                break

            print(
                f"{candidate!r} is not the correct password."
            )

    print("\nExperiment summary")
    print("-" * 70)
    print(f"Model: {model_id}")
    print(f"Input tokens: {total_input_tokens}")
    print(f"Output tokens: {total_output_tokens}")
    print(f"Cost: ${total_cost:.8f}")


def passwords() -> Dict[str, str]:
    """
    Load and validate the passwords extracted from models G, H, and I.
    """

    load_dotenv()

    student_email = os.getenv("STUDENT_EMAIL")
    password_G = os.getenv("PASSWORD_G")
    password_H = os.getenv("PASSWORD_H")
    password_I = os.getenv("PASSWORD_I")

    if not student_email:
        raise ValueError(
            "STUDENT_EMAIL is not set in .env."
        )

    extracted_passwords = {
        "G": password_G,
        "H": password_H,
        "I": password_I,
    }

    for model_id, password in extracted_passwords.items():
        if not password:
            raise ValueError(
                f"PASSWORD_{model_id} is not set in .env. "
                f"Extract and verify the password for model {model_id} first."
            )

        if not test_password(
            password=password,
            model_id=model_id,
            key=student_email,
        ):
            raise ValueError(
                f"PASSWORD_{model_id} is incorrect for "
                f"STUDENT_EMAIL={student_email!r}."
            )

    return {
        "email": student_email,
        "G": password_G,
        "H": password_H,
        "I": password_I,
    }


def print_passwords() -> None:
    # Prints the contents of the passwords function
    p = passwords()
    print(f"Email: {p['email']}")
    print(f"Password G: {p['G']}")
    print(f"Password H: {p['H']}")
    print(f"Password I: {p['I']}")

if __name__=="__main__":

    load_dotenv()

    # Example testing password - this will print False as the password is wrong
    student_email = os.getenv("STUDENT_EMAIL")
    if not student_email:
        raise ValueError("STUDENT_EMAIL environment variable not set. Set it in your environment (or a local .env) to use models G/H/I.")

    print(test_password(
        password="Wrong password",
        model_id="G",
        key=student_email
    ))

    redteam_model("G")
    print()
    redteam_model("H")
    print()
    redteam_model("I")
    print()

    print("Testing password 'hazel' for model G:")
    print(test_password(
        password="hazel",
        model_id="G",
        key=student_email
    ))

    print("Testing password 'ember' for model H:")
    print(test_password(
        password="ember",
        model_id="H",
        key=student_email
    ))

    print("Testing password 'glacier' for model I:")
    print(test_password(
        password="glacier",
        model_id="I",
        key=student_email
    ))

    print_passwords()