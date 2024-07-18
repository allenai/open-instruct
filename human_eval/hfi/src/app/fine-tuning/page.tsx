import Navbar from "../components/navbar";
import Evaluation from "./evaluation";
import Feedback from "./feedback";
import InstructionAndInput from "./instruction-and-input";
import ModelOutput from "./model-output";

export default function Home() {
  return (
    <>
    <Navbar />
    <div className="flex flex-row">
        <div className="mx-auto container">
            <InstructionAndInput />
            <ModelOutput />
            <Evaluation />
        </div>
    </div>

    </>
  );
}
