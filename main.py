import collect_faces
import trainer
import recognizer

def main():
    while True:
        print("\n=== Face Recognition System ===")
        print("1) Collect faces (webcam)")
        print("2) Import faces (folder)")
        print("3) Train model only")
        print("4) Recognize (webcam)")
        print("5) Recognize (image)")
        print("6) Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            collect_faces.capture_from_webcam()
        elif choice == "2":
            collect_faces.import_from_folder()
        elif choice == "3":
            trainer.main()
        elif choice == "4":
            recognizer.recognize_from_webcam()
        elif choice == "5":
            path = input("Enter image file path: ").strip()
            recognizer.recognize_from_image(path)
        elif choice == "6":
            print("Bye!")
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()
