import Navbar from "../components/navbar";
import Evaluation from "./evaluation";
import Feedback from "./feedback";
import InstructionAndInput from "./instruction-and-input";
import ModelOutput from "./model-output";

export default function Home() {
  return (
    <>
    <Navbar />
    <div className="flex flex-row w-full">
        <div className="container flex flex-col lg:flex-row  mx-auto">
            <div className="flex flex-col lg:w-4/6 p-4">
              <InstructionAndInput />
              <ModelOutput />
            </div>
            <div className="flex flex-col lg:w-2/6 p-4">
            <Evaluation />
            </div>
        </div>
    </div>

    </>
  );
}
