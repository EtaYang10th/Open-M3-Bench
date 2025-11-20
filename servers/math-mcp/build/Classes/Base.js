export class Base {
    /**
     * Return a string to the LLM
     * @param value The full string you'd like to return to the LLM
     */
    static formatReturn(value) {
        return {
            content: [{
                    type: "text",
                    text: `${value}`
                }]
        };
    }
}
