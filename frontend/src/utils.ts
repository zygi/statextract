export function stripId(id: string) : string {
    if (id.startsWith('https://openalex.org/')) {
        const res = id.split('/').pop()
        if (!res) {
            throw new Error('Invalid id')
        }
        return res
    }
    return id
}